"""
Mooncake KV Transfer Agent for PD (Prefill-Decode) Separation.

Based official Mooncake TransferEngine API:
https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html

Design principle:
1. Use Mooncake TransferEngine for memory registration and data transfer
2. Use ZMQ for control channel coordination
3. Simplify protocol flow, avoid complex memory management
"""

import json
import struct
import socket
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
import zmq

from nanovllm.config import Config, EngineRole
from nanovllm.engine.sequence import Sequence
from nanovllm.log_config import log_info, log_error, log_warning, log_debug

# Try to import Mooncake
try:
    from mooncake.engine import TransferEngine

    MMOONCAKE_AVAILABLE = True
except ImportError:
    MMOONCAKE_AVAILABLE = False
    log_warning(
        "kv_transfer_mooncake", "Mooncake not available, TCP fallback will be used"
    )


@dataclass
class MooncakeConfig:
    """Mooncake 传输配置"""

    mooncake_master: str = "P2PHANDSHAKE"  # P2P 模式无需 etcd
    local_hostname: str = "localhost"
    transport_protocol: str = "tcp"
    zmq_port: int = 12345


class KVCacheSerializer:
    """KV Cache 序列化器"""

    PROTOCOL_VERSION = 1

    def serialize(
        self, seq_id: int, kv_data: torch.Tensor, block_table: List[int], seq_len: int
    ) -> bytes:
        """
        序列化 KV Cache 数据。

        格式: [metadata_size: 4B][metadata_json][kv_bytes]
        """
        # 转换 BFloat16 为 Float32 以确保兼容性
        if kv_data.dtype == torch.bfloat16:
            kv_data = kv_data.float()

        kv_numpy = kv_data.cpu().numpy()

        # 构建元数据
        metadata = {
            "version": self.PROTOCOL_VERSION,
            "seq_id": seq_id,
            "kv_shape": list(kv_data.shape),
            "kv_dtype": str(kv_numpy.dtype),
            "block_table": block_table,
            "seq_len": seq_len,
        }

        # 序列化为 JSON + 二进制格式
        metadata_json = json.dumps(metadata).encode("utf-8")
        metadata_size = len(metadata_json)

        kv_bytes = kv_numpy.tobytes()
        data = struct.pack("!I", metadata_size) + metadata_json + kv_bytes

        return data

    def deserialize(self, data: bytes) -> Tuple[int, torch.Tensor, List[int], int]:
        """
        反序列化 KV Cache 数据。
        """
        # 解析格式
        metadata_size = struct.unpack("!I", data[:4])[0]
        metadata_json = data[4 : 4 + metadata_size].decode("utf-8")
        metadata = json.loads(metadata_json)

        kv_bytes = data[4 + metadata_size :]

        # 验证版本
        if metadata["version"] != self.PROTOCOL_VERSION:
            raise ValueError(
                f"Protocol version mismatch: expected {self.PROTOCOL_VERSION}"
            )

        # 重建数组
        kv_shape = tuple(metadata["kv_shape"])
        kv_dtype = np.dtype(metadata["kv_dtype"])
        kv_numpy = np.frombuffer(kv_bytes, dtype=kv_dtype).reshape(kv_shape)

        # 转换为 torch tensor
        kv_data = torch.from_numpy(kv_numpy)

        return (
            metadata["seq_id"],
            kv_data,
            metadata["block_table"],
            metadata["seq_len"],
        )

    def serialize_batch(self, kv_dict: Dict[int, Dict[str, Any]]) -> bytes:
        """批量序列化"""
        items_data = {}
        items_bytes = b""

        for seq_id, data in kv_dict.items():
            serialized = self.serialize(
                seq_id, data["kv_data"], data["block_table"], data["seq_len"]
            )
            items_data[seq_id] = {
                "offset": len(items_bytes),
                "length": len(serialized),
            }
            items_bytes += serialized

        batch_metadata = {
            "version": self.PROTOCOL_VERSION,
            "count": len(kv_dict),
            "items": items_data,
        }

        metadata_json = json.dumps(batch_metadata).encode("utf-8")
        metadata_size = len(metadata_json)

        return struct.pack("!I", metadata_size) + metadata_json + items_bytes

    def deserialize_batch(self, data: bytes) -> Dict[int, Dict[str, Any]]:
        """批量反序列化"""
        metadata_size = struct.unpack("!I", data[:4])[0]
        metadata_json = data[4 : 4 + metadata_size].decode("utf-8")
        batch_metadata = json.loads(metadata_json)

        data_offset = 4 + metadata_size
        items_info = batch_metadata["items"]

        result = {}
        for seq_id_str, info in items_info.items():
            seq_id = int(seq_id_str)
            offset = info["offset"]
            length = info["length"]
            item_data = data[data_offset + offset : data_offset + offset + length]

            seq_id_kv, kv_data, block_table, seq_len = self.deserialize(item_data)
            assert seq_id_kv == seq_id

            result[seq_id] = {
                "kv_data": kv_data,
                "block_table": block_table,
                "seq_len": seq_len,
            }

        return result


class MooncakeTransferAgent:
    """
    基于 Mooncake TransferEngine 的 KV Cache 传输代理。

    工作流程:
    1. Prefill 节点完成 prefill 后，序列化 KV cache
    2. Prefill 节点通过 ZMQ 通知 Decode 节点数据大小
    3. Decode 节点准备接收缓冲区
    4. Prefill 节点发送数据（通过 ZMQ 或 Mooncake）
    5. Decode 节点接收并反序列化数据
    """

    def __init__(self, config: Config, scheduler: Any):
        self.config = config
        self.scheduler = scheduler
        self.role = config.engine_role
        self.address = config.kv_transfer_address
        self.port = config.kv_transfer_port

        self.serializer = KVCacheSerializer()
        self.transfer_engine = None
        self.session_id = None
        self._initialized = False

        # ZMQ 控制通道
        self.zmq_context = None
        self.zmq_socket = None

        # 统计
        self.stats = {
            "sent_bytes": 0,
            "received_bytes": 0,
            "sent_count": 0,
            "received_count": 0,
            "errors": 0,
        }

        # 初始化
        if MMOONCAKE_AVAILABLE:
            self._init_mooncake()
        else:
            log_warning(
                "kv_transfer_mooncake", "Mooncake not available, using TCP fallback"
            )
            self._init_tcp()

    def _init_mooncake(self):
        """初始化 Mooncake TransferEngine"""
        try:
            self.transfer_engine = TransferEngine()

            # 使用位置参数（官方 API 要求）
            result = self.transfer_engine.initialize(
                self.address,  # hostname
                "P2PHANDSHAKE",  # metadata_server - P2P 模式
                "tcp",  # protocol
                "",  # device_name
            )

            if result < 0:
                log_error(
                    "kv_transfer_mooncake",
                    f"Failed to initialize TransferEngine: {result}",
                )
                self._init_tcp()
                return

            self.session_id = f"{self.address}:{self.transfer_engine.get_rpc_port()}"

            log_info(
                "kv_transfer_mooncake",
                f"Mooncake TransferEngine initialized, session_id: {self.session_id}",
            )

            self._init_zmq_control()
            self._initialized = True

        except Exception as e:
            log_error("kv_transfer_mooncake", f"Failed to initialize Mooncake: {e}")
            self._init_tcp()

    def _init_tcp(self):
        """TCP 回退模式初始化"""
        self.transfer_engine = None
        self._initialized = True
        log_info("kv_transfer_mooncake", "Using TCP fallback mode")
        self._init_zmq_control()

    def _init_zmq_control(self):
        """初始化 ZMQ 控制通道"""
        self.zmq_context = zmq.Context()

        if self.role == EngineRole.DECODE:
            # Decode 节点作为服务器
            self.zmq_socket = self.zmq_context.socket(zmq.REP)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            # 使用 RANDOM_PORT 让系统自动分配可用端口
            self.zmq_socket.bind(f"tcp://*:{self.port}")
            log_info(
                "kv_transfer_mooncake",
                f"ZMQ control server listening on port {self.port}",
            )
        else:
            # Prefill 节点作为客户端 - 不在这里等待连接
            self.zmq_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.setsockopt(zmq.CONNECT_TIMEOUT, 10000)  # 10秒连接超时
            self.zmq_socket.connect(f"tcp://{self.address}:{self.port}")
            log_info(
                "kv_transfer_mooncake",
                f"ZMQ control client connecting to {self.address}:{self.port}",
            )

    def send_sequences(
        self, seqs: List[Sequence], kv_data: Dict[int, torch.Tensor]
    ) -> bool:
        """
        Prefill 节点发送序列和 KV Cache 到 Decode 节点。
        """
        if not seqs or not kv_data:
            return True

        if not self._initialized:
            log_error("kv_transfer_mooncake", "Transfer agent not initialized")
            return False

        try:
            # 准备数据
            kv_dict = {}
            for seq in seqs:
                if seq.seq_id in kv_data:
                    kv_dict[seq.seq_id] = {
                        "kv_data": kv_data[seq.seq_id],
                        "block_table": seq.block_table,
                        "seq_len": len(seq),
                    }

            if not kv_dict:
                return True

            # 序列化
            serialized_data = self.serializer.serialize_batch(kv_dict)
            data_size = len(serialized_data)

            log_debug(
                "kv_transfer_mooncake",
                f"Sending {len(kv_dict)} sequences, total size: {data_size / 1024 / 1024:.2f} MB",
            )

            # 设置超时
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, 30000)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, 30000)

            # 发送传输请求
            request = {
                "session_id": self.session_id,
                "length": data_size,
            }
            self.zmq_socket.send_json(request)

            # 接收确认
            response = self.zmq_socket.recv_json()

            # 发送数据
            self.zmq_socket.send(serialized_data)

            self.stats["sent_bytes"] += data_size
            self.stats["sent_count"] += len(kv_dict)

            log_debug(
                "kv_transfer_mooncake", f"Successfully sent {len(kv_dict)} sequences"
            )
            return True

        except Exception as e:
            self.stats["errors"] += 1
            log_error(f"Error sending sequences: {e}")
            return False

    def recv_sequences(self) -> List[Tuple[Sequence, torch.Tensor]]:
        """
        Decode 节点接收序列和 KV Cache。
        """
        if not self._initialized or self.role != EngineRole.DECODE:
            return []

        received_items = []

        try:
            # 设置非阻塞接收
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms 超时

            try:
                request = self.zmq_socket.recv_json(zmq.NOBLOCK)
            except zmq.error.Again:
                return received_items

            log_debug("kv_transfer_mooncake", f"Received transfer request: {request}")

            data_size = request["length"]

            # 发送确认
            response = {"status": "ready", "length": data_size}
            self.zmq_socket.send_json(response)

            # 接收数据
            serialized_data = self.zmq_socket.recv()

            # 反序列化
            kv_dict = self.serializer.deserialize_batch(serialized_data)

            for seq_id, data_dict in kv_dict.items():
                seq = self._create_sequence(seq_id, data_dict)
                if seq is not None:
                    received_items.append((seq, data_dict["kv_data"]))

            self.stats["received_bytes"] += len(serialized_data)
            self.stats["received_count"] += len(kv_dict)

            log_debug("kv_transfer_mooncake", f"Received {len(kv_dict)} sequences")

        except Exception as e:
            self.stats["errors"] += 1
            log_error("kv_transfer_mooncake", f"Error receiving sequences: {e}")

        return received_items

    def _create_sequence(
        self, seq_id: int, data_dict: Dict[str, Any]
    ) -> Optional[Sequence]:
        """从接收到的数据创建 Sequence 对象"""
        # 这里需要根据实际协议完善
        # 返回 None 表示需要上层逻辑创建 Sequence
        log_warning("kv_transfer_mooncake", "_create_sequence needs implementation")
        return None

    def close(self):
        """关闭传输代理"""
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        if self.transfer_engine:
            # Mooncake TransferEngine 清理
            pass

        log_info("kv_transfer_mooncake", "Transfer agent closed")

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats
