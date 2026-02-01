import unittest
from unittest.mock import MagicMock
from nanovllm.config import Config, EngineRole
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus

class TestPDBatchSize(unittest.TestCase):
    def setUp(self):
        # 创建基础配置
        self.model_path = "/tmp/dummy_model"
        import os
        os.makedirs(self.model_path, exist_ok=True)
        
        # 准备 Mock AutoConfig
        from transformers import AutoConfig
        self.hf_config = MagicMock()
        self.hf_config.max_position_embeddings = 4096
        self.hf_config.num_hidden_layers = 2
        self.hf_config.num_attention_heads = 8
        self.hf_config.hidden_size = 512
        self.hf_config.dtype = "float16"
        
        # 为了让 Config.__post_init__ 跑通，我们需要 Mock AutoConfig.from_pretrained
        import transformers
        AutoConfig.from_pretrained = MagicMock(return_value=self.hf_config)

    def test_separate_batch_sizes(self):
        # 设置不同的 Prefill 和 Decode batch size
        config = Config(
            model=self.model_path,
            max_num_seqs=10,
            max_num_seqs_prefill=2,
            max_num_seqs_decode=5,
            num_kvcache_blocks=1000,
            kvcache_block_size=256
        )
        
        scheduler = Scheduler(config)
        
        # 1. 测试 Prefill Batch Size
        # 添加 5 个请求
        for i in range(5):
            seq = Sequence([1, 2, 3])
            scheduler.add(seq)
            
        # 第一轮调度，角色默认为 SINGLE (逻辑上 Prefill 阶段会先跑)
        # 但我们之前修改了 Scheduler，它在 schedule() 中会根据 config.engine_role 行为有所不同
        # 即使是 SINGLE 模式，它也会先尝试 Prefill
        
        # 模拟 Prefill 节点
        scheduler.config.engine_role = EngineRole.PREFILL
        scheduled_seqs, is_prefill = scheduler.schedule()
        
        self.assertTrue(is_prefill)
        self.assertEqual(len(scheduled_seqs), 2) # 应该受限于 max_num_seqs_prefill=2
        
        # 2. 测试 Decode Batch Size
        # 模拟 Decode 节点
        scheduler.config.engine_role = EngineRole.DECODE
        
        # 手动往 running 队列塞 10 个 seq
        for i in range(10):
            seq = Sequence([1, 2, 3])
            seq.status = SequenceStatus.RUNNING
            # 必须分配 blocks 才能进入 decode
            scheduler.block_manager.allocate(seq)
            scheduler.running.append(seq)
            
        scheduled_seqs, is_prefill = scheduler.schedule()
        
        self.assertFalse(is_prefill)
        self.assertEqual(len(scheduled_seqs), 5) # 应该受限于 max_num_seqs_decode=5

    def test_fallback_to_default(self):
        # 测试如果没有指定，是否回退到 max_num_seqs
        config = Config(
            model=self.model_path,
            max_num_seqs=8,
            max_num_seqs_prefill=None,
            max_num_seqs_decode=None,
            num_kvcache_blocks=1000,
            kvcache_block_size=256
        )
        
        self.assertEqual(config.max_num_seqs_prefill, 8)
        self.assertEqual(config.max_num_seqs_decode, 8)
        
        scheduler = Scheduler(config)
        self.assertEqual(scheduler.max_num_seqs_prefill, 8)
        self.assertEqual(scheduler.max_num_seqs_decode, 8)

if __name__ == "__main__":
    unittest.main()
