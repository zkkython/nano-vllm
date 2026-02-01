import socket
import struct
import torch
import io
import time
from typing import List, Tuple, Any
from nanovllm.config import Config, EngineRole
from nanovllm.engine.sequence import Sequence
from nanovllm.log_config import log_info, log_error

class KVTransferAgent:
    def __init__(self, config: Config, scheduler: Any):
        self.config = config
        self.scheduler = scheduler
        self.role = config.engine_role
        self.port = config.kv_transfer_port
        self.address = config.kv_transfer_address
        
        if self.role == EngineRole.DECODE:
            # Decode node acts as a server receiving from Prefill nodes
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind(('', self.port))
            self.server_sock.listen(128)
            self.server_sock.setblocking(False)
            log_info("kv_transfer", f"Decode node listening on port {self.port}")

    def send_sequences(self, seqs: List[Sequence], kv_data: dict):
        """
        Prefill node sends sequences and their KV data to Decode node.
        kv_data: seq_id -> KV tensor
        """
        if not seqs:
            return
            
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create a connection for this transfer
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(60.0)  # 增加超时时间以支持大数据量传输
                    s.connect((self.address, self.port))
                    
                    # Prepare payload
                    payload = []
                    for seq in seqs:
                        if seq.seq_id in kv_data:
                            payload.append((seq, kv_data[seq.seq_id]))
                    
                    if not payload:
                        return

                    # 使用内存视图发送，减少拷贝
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    data_view = buffer.getbuffer()
                    
                    # 发送长度前缀和数据
                    s.sendall(struct.pack('!Q', len(data_view)))
                    s.sendall(data_view)
                    log_info("kv_transfer", f"Successfully sent {len(payload)} sequences ({len(data_view)/1024/1024:.2f} MB) to {self.address}:{self.port}")
                    return # 成功后退出循环
            except (ConnectionRefusedError, socket.timeout, BrokenPipeError) as e:
                if attempt < max_retries - 1:
                    log_error("kv_transfer", f"Connection/Transfer failed (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...: {e}")
                    time.sleep(retry_delay)
                else:
                    log_error("kv_transfer", f"Failed to send sequences to {self.address}:{self.port} after {max_retries} attempts: {e}")
            except Exception as e:
                log_error("kv_transfer", f"Failed to send sequences to {self.address}:{self.port}: {e}")
                break

    def recv_sequences(self) -> List[Tuple[Sequence, torch.Tensor]]:
        """
        Decode node receives sequences and their KV data.
        Returns a list of (Sequence, KV_tensor)
        """
        if self.role != EngineRole.DECODE:
            return []
            
        received_items = []
        while True:
            try:
                conn, addr = self.server_sock.accept()
            except BlockingIOError:
                # 没有新的连接
                break
            except Exception as e:
                log_error("kv_transfer", f"Accept error: {e}")
                break
            
            try:
                with conn:
                    conn.settimeout(60.0) # 接收端也需要较长的超时
                    # 读取长度前缀
                    size_data = self._read_n_bytes(conn, 8)
                    if not size_data:
                        continue
                    size = struct.unpack('!Q', size_data)[0]
                    
                    # 读取数据负载
                    payload_data = self._read_n_bytes(conn, size)
                    if not payload_data:
                        continue
                    
                    buffer = io.BytesIO(payload_data)
                    # 允许 pickle 加载
                    items = torch.load(buffer, weights_only=False, map_location='cpu')
                    received_items.extend(items)
                    log_info("kv_transfer", f"Successfully received {len(items)} sequences ({size/1024/1024:.2f} MB) from {addr}")
            except Exception as e:
                log_error("kv_transfer", f"Error processing connection from {addr}: {e}")
            
        return received_items

    def _read_n_bytes(self, conn, n):
        """高效读取指定字节数"""
        view = memoryview(bytearray(n))
        pos = 0
        while pos < n:
            nread = conn.recv_into(view[pos:], n - pos)
            if nread == 0:
                return None
            pos += nread
        return view.tobytes()
