from dataclasses import dataclass
import os
from glob import glob
import logging

import torch
from torch import nn
from tqdm import tqdm
from safetensors import safe_open

logger = logging.getLogger(__name__)


@dataclass
class WeightMapping:

    target_path: str | list[str]
    transpose: bool = False
    dist_strategy: str | None = None
    loader_arg: object | None = None


class WeightLoader:
    def __init__(self, model: nn.Module, config, model_path: str):
        self.model_path = model_path
        self.model_config = config
        self.model = model

    def load_weights_from_safetensors(
        self, weight_mappings: dict[str, str | list[str] | WeightMapping]
    ):
        # 标准化映射配置，统一成 WeightMapping 结构
        normalized_mappings: dict[str, WeightMapping] = {}
        for src_key, mapping in weight_mappings.items():
            if isinstance(mapping, WeightMapping):
                normalized_mappings[src_key] = mapping
            else:
                normalized_mappings[src_key] = WeightMapping(target_path=mapping)

        # 记录模型的所有参数，方便通过字符串路径查找
        param_dict: dict[str, nn.Parameter] = dict(self.model.named_parameters())

        def match_mapping(hf_key: str) -> tuple[WeightMapping | None, dict]:
            """根据权重名找到对应的 WeightMapping 和格式化参数.

            支持两种形式：
            1. 精确匹配：hf_key 直接在 normalized_mappings 中
            2. 模板匹配：key 中包含 "{layer}", 例如
               "model.layers.{layer}.input_layernorm.weight"
            """

            # 1. 精确匹配
            mapping = normalized_mappings.get(hf_key)
            if mapping is not None:
                return mapping, {}

            # 2. 按层号的模板匹配
            for pattern, m in normalized_mappings.items():
                if "{layer}" not in pattern:
                    continue

                prefix, suffix = pattern.split("{layer}", 1)
                if not (hf_key.startswith(prefix) and hf_key.endswith(suffix)):
                    continue

                middle = hf_key[len(prefix) : len(hf_key) - len(suffix)]
                if not middle.isdigit():
                    continue

                layer_idx = int(middle)
                return m, {"layer": layer_idx}

            return None, {}

        for hf_key, weight_tensor in self._iterate_weights():
            mapping, fmt_kwargs = match_mapping(hf_key)

            # 如果没有提供显式映射，则默认源名 == 目标名
            if mapping is None:
                target_paths: list[str] = [hf_key]
                transpose = False
            else:
                if isinstance(mapping.target_path, str):
                    target_paths = [mapping.target_path]
                else:
                    target_paths = list(mapping.target_path)
                transpose = mapping.transpose

            tensor = weight_tensor
            if transpose:
                tensor = tensor.T

            for target_path in target_paths:
                # 支持 target_path 中使用 {layer} 之类的占位符
                if fmt_kwargs:
                    try:
                        target_name = target_path.format(**fmt_kwargs)
                    except Exception:
                        target_name = target_path
                else:
                    target_name = target_path

                param = param_dict.get(target_name)
                if param is None:
                    logger.warning(
                        "Parameter %s not found in model when loading %s, skipping",
                        target_name,
                        hf_key,
                    )
                    continue

                # 如果参数上挂了自定义 weight_loader（例如并行 Linear），优先走自定义逻辑
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    if mapping is not None and mapping.loader_arg is not None:
                        weight_loader(param, tensor, mapping.loader_arg)
                    else:
                        weight_loader(param, tensor)
                else:
                    if tensor.shape != param.shape:
                        logger.warning(
                            "Shape mismatch for %s (from %s): tensor %s vs param %s, skipping",
                            target_name,
                            hf_key,
                            tuple(tensor.shape),
                            tuple(param.shape),
                        )
                        continue

                    param.data.copy_(tensor.to(param.device, dtype=param.dtype))

    def _iterate_weights(self):
        model_path = self.model_path
        weights_files = glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {model_path}")

        weights_files.sort()

        skipped_files = 0
        with tqdm(weights_files, desc="[LOADING] MODEL WEIGHTS", unit="file") as pbar:
            for st_file in pbar:
                filename = os.path.basename(st_file)
                pbar.set_postfix({"file": filename})

                with (safe_open(st_file, "pt", "cpu") as f,):
                    needed_keys = []
                    for name in f.keys():  # noqa: SIM118
                        if not name.startswith("model.layers."):
                            needed_keys.append(name)
                            continue

                        if not self._is_excluded_layer_weight(name):
                            needed_keys.append(name)

                    if not needed_keys:
                        skipped_files += 1
                        logger.debug(
                            "Skipping %s: 0/%s weights needed",
                            filename,
                            len(f.keys()),
                        )
                        continue

                    logger.debug(
                        "Loading %s: %s/%s weights needed",
                        filename,
                        len(needed_keys),
                        len(f.keys()),
                    )
                    for name in needed_keys:
                        weight_tensor = f.get_tensor(name)
                        # 对外输出name 以及  加载的weight_tensor
                        yield name, weight_tensor

        if skipped_files > 0:
            logger.info(
                "Memory optimization: Skipped %s/%s files with no needed weights",
                skipped_files,
                len(weights_files),
            )

    def _is_excluded_layer_weight(self, hf_key: str) -> bool:
        if not hf_key.startswith("model.layers."):
            return False

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return False

        layer_num = int(parts[2])

        is_excluded = layer_num >= self.model_config.num_hidden_layers

        if is_excluded and not hasattr(self, "_debug_count"):
            logger.info(
                "DEBUG: Excluding layer %s >= %s",
                layer_num,
                self.model_config.num_hidden_layers,
            )
            self._debug_count = True

        return is_excluded
