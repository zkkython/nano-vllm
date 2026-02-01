"""
测试 Mooncake KV Transfer 模块
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.engine.kv_transfer_mooncake import (
    KVCacheSerializer,
    MooncakeTransferAgent,
)
from nanovllm.config import Config, EngineRole
from unittest.mock import Mock


def test_serializer():
    """测试 KV Cache 序列化器"""
    print("Testing KVCacheSerializer...")

    serializer = KVCacheSerializer()

    # 创建测试数据
    kv_data = torch.randn(32, 128, 8)  # (num_layers, seq_len, hidden_size)
    block_table = [0, 1, 2, 3]
    seq_len = 128
    seq_id = 1001

    # 序列化
    serialized = serializer.serialize(seq_id, kv_data, block_table, seq_len)
    print(f"  Serialized size: {len(serialized)} bytes")

    # 反序列化
    recv_seq_id, recv_kv_data, recv_block_table, recv_seq_len = serializer.deserialize(
        serialized
    )

    # 验证
    assert recv_seq_id == seq_id
    assert recv_seq_len == seq_len
    assert recv_block_table == block_table
    assert list(recv_kv_data.shape) == [32, 128, 8]

    print("  ✓ Serialization test passed")


def test_batch_serializer():
    """测试批量序列化"""
    print("Testing batch serialization...")

    serializer = KVCacheSerializer()

    kv_dict = {
        1001: {
            "kv_data": torch.randn(32, 128, 8),
            "block_table": [0, 1, 2, 3],
            "seq_len": 128,
        },
        1002: {
            "kv_data": torch.randn(32, 64, 8),
            "block_table": [4, 5, 6],
            "seq_len": 64,
        },
    }

    # 批量序列化
    serialized = serializer.serialize_batch(kv_dict)
    print(f"  Batch serialized size: {len(serialized)} bytes")

    # 批量反序列化
    deserialized = serializer.deserialize_batch(serialized)

    # 验证
    assert len(deserialized) == 2
    assert 1001 in deserialized
    assert 1002 in deserialized

    print("  ✓ Batch serialization test passed")


def test_transfer_agent_init():
    """测试传输代理初始化"""
    print("Testing MooncakeTransferAgent initialization...")

    # 创建 mock config
    mock_config = Mock()
    mock_config.engine_role = EngineRole.DECODE
    mock_config.kv_transfer_port = 23456
    mock_config.kv_transfer_address = "localhost"

    mock_scheduler = Mock()

    # 创建代理
    agent = MooncakeTransferAgent(mock_config, mock_scheduler)

    assert agent._initialized
    assert agent.role == EngineRole.DECODE
    assert agent.port == 23456

    # 关闭
    agent.close()

    print("  ✓ Transfer agent initialization test passed")


def test_transfer_agent_tcp_mode():
    """测试 TCP 模式传输"""
    print("Testing TCP mode transfer...")

    import zmq

    # 创建两个代理模拟通信
    context = zmq.Context()

    # Decode 节点
    decode_config = Mock()
    decode_config.engine_role = EngineRole.DECODE
    decode_config.kv_transfer_port = 23456
    decode_config.kv_transfer_address = "localhost"

    decode_scheduler = Mock()
    decode_agent = MooncakeTransferAgent(decode_config, decode_scheduler)

    # Prefill 节点
    prefill_config = Mock()
    prefill_config.engine_role = EngineRole.PREFILL
    prefill_config.kv_transfer_port = 23457
    prefill_config.kv_transfer_address = "localhost"

    prefill_scheduler = Mock()
    prefill_agent = MooncakeTransferAgent(prefill_config, prefill_scheduler)

    # 验证两个节点都已初始化
    assert decode_agent._initialized
    assert prefill_agent._initialized

    # 清理
    decode_agent.close()
    prefill_agent.close()
    context.term()

    print("  ✓ TCP mode transfer test passed")


def main():
    print("=" * 50)
    print("Mooncake KV Transfer Tests")
    print("=" * 50)

    try:
        test_serializer()
        test_batch_serializer()
        test_transfer_agent_init()
        test_transfer_agent_tcp_mode()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
