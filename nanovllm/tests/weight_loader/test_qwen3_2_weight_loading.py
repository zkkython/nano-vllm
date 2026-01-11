import os
from glob import glob

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config

from nanovllm.models.qwen3_2 import Qwen3ForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

"""
PYTHONPATH=/root/mingtong/aiwork/nano-vllm pytest nanovllm/tests/weight_loader/test_qwen3_2_weight_loading.py

PYTHONPATH=/root/mingtong/aiwork/nano-vllm pytest nanovllm/tests/weight_loader/test_qwen3_2_weight_loading.py::test_qwen3_2_load_weights -q
PYTHONPATH=/root/mingtong/aiwork/nano-vllm pytest nanovllm/tests/weight_loader/test_qwen3_2_weight_loading.py::test_qwen3_2_load_real_qwen3_weights -q

"""


def _patch_dist_single_process():
    """Patch torch.distributed for single-process tests.

    这里简单把 world_size 固定为 1，rank 固定为 0，all_reduce 变成 no-op，
    方便在未初始化分布式环境下测试并行 Linear 的权重加载逻辑。
    """

    import torch.distributed as dist

    dist.get_world_size = lambda: 1
    dist.get_rank = lambda: 0

    def _noop_all_reduce(tensor):  # noqa: ANN001
        return tensor

    dist.all_reduce = _noop_all_reduce


def _get_hf_tensor(weights_files: list[str], key: str) -> torch.Tensor:
    """在多个 safetensors shard 中查找指定 key 对应的 tensor."""

    for st_file in weights_files:
        with safe_open(st_file, "pt", "cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f"Key {key} not found in any weights file")


def test_qwen3_2_load_weights(tmp_path):  # noqa: D103
    _patch_dist_single_process()

    # 构造一个小尺寸的 Qwen3 配置，便于在 CPU 上快速测试
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )

    model = Qwen3ForCausalLM(config)

    # 从模型内部构建映射规则
    weight_mappings = model._build_weight_mappings()

    # 根据映射规则和目标参数形状，构造一份假的 HF 权重字典
    param_dict = dict(model.named_parameters())
    tensors: dict[str, torch.Tensor] = {}

    for hf_key, mapping in weight_mappings.items():
        if isinstance(mapping, str):
            target_path = mapping
            transpose_flag = False
        else:
            target_path = mapping.target_path  # 当前实现中都是 str
            transpose_flag = getattr(mapping, "transpose", False)

        param = param_dict.get(target_path)
        if param is None:
            continue

        shape = param.shape
        # 对于需要 transpose 的权重，假定 HF 端形状为 (out, in)，模型端为 (in, out)
        if transpose_flag and param.dim() == 2:
            hf_shape = (shape[1], shape[0])
        else:
            hf_shape = shape

        # 用可预测的数据填充，方便后面做精确断言
        numel = hf_shape[0] * hf_shape[1] if len(hf_shape) == 2 else param.numel()
        tensors[hf_key] = torch.arange(numel, dtype=torch.float32).reshape(hf_shape)

    # 将构造好的权重保存为单个 safetensors 文件
    model_dir = tmp_path / "qwen3_fake"
    os.makedirs(model_dir, exist_ok=True)
    weights_file = model_dir / "model-00001-of-00001.safetensors"
    save_file(tensors, str(weights_file))

    # 调用模型自带的 load_weights，从 safetensors 中加载权重
    model.load_weights(config=config, model_path=str(model_dir))

    # === 验证若干关键权重是否按预期被加载 ===
    # 1) 词嵌入权重：直接映射，无并行切分
    embed_param = model.transformers.embed_tokens.weight
    assert torch.allclose(
        embed_param,
        tensors["model.embed_tokens.weight"].to(embed_param.dtype),
    )

    # 2) 第一层 input_layernorm：一维权重，直接拷贝
    ln_param = model.transformers.layers[0].input_layernorm.weight
    assert torch.allclose(
        ln_param,
        tensors["model.layers.0.input_layernorm.weight"].to(ln_param.dtype),
    )

    # 3) 第一层 self_attn.q_proj：列并行 + 转置，但在 world_size=1 下等价于简单转置
    hf_q_key = "model.layers.0.self_attn.q_proj.weight"
    q_target_name = weight_mappings[hf_q_key].target_path
    q_param = param_dict[q_target_name]
    expected_q = tensors[hf_q_key].T.to(q_param.dtype)
    assert torch.allclose(q_param, expected_q)

    # 4) MLP down_proj：行并行 + 转置，在 world_size=1 下应等价于简单转置
    hf_down_key = "model.layers.0.mlp.down_proj.weight"
    down_target_name = weight_mappings[hf_down_key].target_path
    down_param = param_dict[down_target_name]
    expected_down = tensors[hf_down_key].T.to(down_param.dtype)
    assert torch.allclose(down_param, expected_down)


@pytest.mark.skipif(
    not os.path.isdir(
        os.environ.get(
            "QWEN3_MODEL_PATH", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
        )
    ),
    reason="Real Qwen3 model path not found; set QWEN3_MODEL_PATH to enable this test.",
)
def test_qwen3_2_load_real_qwen3_weights():  # noqa: D103
    _patch_dist_single_process()

    model_path = os.environ.get(
        "QWEN3_MODEL_PATH", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    )

    # 从真实模型目录读取配置
    config = Qwen3Config.from_pretrained(model_path)
    model = Qwen3ForCausalLM(config)

    # 先加载权重
    model.load_weights(config=config, model_path=model_path)

    # 找到所有 safetensors shard，方便读取 HF 权重
    weights_files = glob(os.path.join(model_path, "*.safetensors"))
    assert weights_files, "No *.safetensors files found in real Qwen3 model path"

    # 1) 验证 embedding 权重
    embed_src = _get_hf_tensor(weights_files, "model.embed_tokens.weight")
    embed_param = model.transformers.embed_tokens.weight
    assert embed_src.shape == embed_param.shape
    assert torch.allclose(embed_param, embed_src.to(embed_param.dtype))

    # 2) 第一层 input_layernorm
    ln_src = _get_hf_tensor(weights_files, "model.layers.0.input_layernorm.weight")
    ln_param = model.transformers.layers[0].input_layernorm.weight
    assert ln_src.shape == ln_param.shape
    assert torch.allclose(ln_param, ln_src.to(ln_param.dtype))

    # 3) 第一层 self_attn.q_proj：列并行 + transpose=True，world_size=1 等价于转置
    q_src = _get_hf_tensor(weights_files, "model.layers.0.self_attn.q_proj.weight")
    weight_mappings = model._build_weight_mappings()
    hf_q_key = "model.layers.0.self_attn.q_proj.weight"
    q_target_name = weight_mappings[hf_q_key].target_path
    q_param = dict(model.named_parameters())[q_target_name]
    expected_q = q_src.T.to(q_param.dtype)
    assert q_param.shape == expected_q.shape
    assert torch.allclose(q_param, expected_q)

    # 4) 第一层 MLP down_proj：行并行 + transpose=True，world_size=1 等价于转置
    down_src = _get_hf_tensor(weights_files, "model.layers.0.mlp.down_proj.weight")
    hf_down_key = "model.layers.0.mlp.down_proj.weight"
    down_target_name = weight_mappings[hf_down_key].target_path
    down_param = dict(model.named_parameters())[down_target_name]
    expected_down = down_src.T.to(down_param.dtype)
    assert down_param.shape == expected_down.shape
    assert torch.allclose(down_param, expected_down)


@pytest.mark.skipif(
    not os.path.isdir(
        os.environ.get(
            "QWEN3_MODEL_PATH", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
        )
    ),
    reason="Real Qwen3 model path not found; set QWEN3_MODEL_PATH to enable this test.",
)
def test_qwen3_2_load_real_qwen3_weights_with_layers():  # noqa: D103
    _patch_dist_single_process()

    model_path = os.environ.get(
        "QWEN3_MODEL_PATH", "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    )

    # 从真实模型目录读取配置
    config = Qwen3Config.from_pretrained(model_path)
    model = Qwen3ForCausalLM(config)

    # 先加载权重
    results = model.load_weights(
        config=config, model_path=model_path, load_partial_layers=2
    )
    log.info(results)
    assert results is not None
