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
    def __init__(
        self,
        model: nn.Module,
        config,
        model_path: str,
        load_partial_layers: int | None = None,
    ):
        self.model_path = model_path
        self.model_config = config
        self.model = model
        self.load_partial_layers = load_partial_layers

    def load_weights_from_safetensors(
        self, weight_mappings: dict[str, str | list[str] | WeightMapping]
    ) -> dict[str, int]:
        """从 safetensors 文件中加载权重.

        Args:
            weight_mappings: 权重映射字典

        Returns:
            加载统计信息字典，包含：
            - "total_weights": 总权重数量
            - "loaded_weights": 成功加载的权重数量
            - "skipped_weights": 跳过的权重数量（超出层数或部分加载）
            - "loaded_layers": 实际加载的层号列表
            - "skipped_layers": 跳过的层号列表
            - "loaded_weight_names": 已加载的权重名称列表
            - "skipped_weight_names": 跳过的权重名称列表
        """
        # 标准化映射配置，统一成 WeightMapping 结构
        normalized_mappings: dict[str, WeightMapping] = {}
        for src_key, mapping in weight_mappings.items():
            if isinstance(mapping, WeightMapping):
                normalized_mappings[src_key] = mapping
            else:
                normalized_mappings[src_key] = WeightMapping(target_path=mapping)

        # 记录模型的所有参数，方便通过字符串路径查找
        param_dict: dict[str, nn.Parameter] = dict(self.model.named_parameters())

        # 统计信息
        stats = {
            "total_weights": 0,
            "loaded_weights": 0,
            "skipped_weights": 0,
            "loaded_layers": set(),
            "skipped_layers": set(),
            "loaded_weight_names": [],
            "skipped_weight_names": [],
        }

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
            stats["total_weights"] += 1

            # 提取层号用于统计
            layer_num = self._extract_layer_num(hf_key)

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

            loaded_any = False
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
                    loaded_any = True
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
                    loaded_any = True

            # 更新统计信息
            if loaded_any:
                stats["loaded_weights"] += 1
                stats["loaded_weight_names"].append(hf_key)
                if layer_num is not None:
                    stats["loaded_layers"].add(layer_num)
            else:
                stats["skipped_weights"] += 1
                stats["skipped_weight_names"].append(hf_key)
                if layer_num is not None:
                    stats["skipped_layers"].add(layer_num)

        # 统计被排除的层和权重（这些层的权重在 _iterate_weights 中被过滤掉了）
        # 需要扫描权重文件来找出被排除的层和权重名称
        excluded_info = self._get_excluded_info()
        for layer_num in excluded_info["layers"]:
            if layer_num not in stats["loaded_layers"]:
                stats["skipped_layers"].add(layer_num)

        # 添加被排除的权重名称和计数
        stats["skipped_weight_names"].extend(excluded_info["weight_names"])
        stats["skipped_weights"] += len(excluded_info["weight_names"])
        stats["total_weights"] += len(excluded_info["weight_names"])

        # 转换 set 为排序后的列表
        stats["loaded_layers"] = sorted(list(stats["loaded_layers"]))
        stats["skipped_layers"] = sorted(list(stats["skipped_layers"]))

        # 输出加载摘要
        logger.info(
            "Weight loading summary: %d/%d weights loaded, %d skipped",
            stats["loaded_weights"],
            stats["total_weights"],
            stats["skipped_weights"],
        )
        if stats["loaded_layers"]:
            logger.info("Loaded layers: %s", stats["loaded_layers"])
        if stats["skipped_layers"]:
            logger.info("Skipped layers: %s", stats["skipped_layers"])

        return stats

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

        # 1. 检查是否超出模型定义的层数
        is_beyond_model = layer_num >= self.model_config.num_hidden_layers

        # 2. 检查是否超出用户指定的部分加载层数
        is_beyond_partial = (
            self.load_partial_layers is not None
            and layer_num >= self.load_partial_layers
        )

        is_excluded = is_beyond_model or is_beyond_partial

        if is_excluded and not hasattr(self, "_debug_count"):
            if is_beyond_model:
                logger.info(
                    "Excluding layer %s >= model layers %s",
                    layer_num,
                    self.model_config.num_hidden_layers,
                )
            elif is_beyond_partial:
                logger.info(
                    "Partial loading: Excluding layer %s >= load_partial_layers %s",
                    layer_num,
                    self.load_partial_layers,
                )
            self._debug_count = True

        return is_excluded

    def _extract_layer_num(self, hf_key: str) -> int | None:
        """从权重 key 中提取层号.

        Args:
            hf_key: HF 权重名，例如 "model.layers.3.self_attn.q_proj.weight"

        Returns:
            层号（int），如果不是层权重则返回 None
        """
        if not hf_key.startswith("model.layers."):
            return None

        parts = hf_key.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            return None

        return int(parts[2])

    def _get_excluded_info(self) -> dict[str, list | set]:
        """获取被排除的层号和权重名称.

        这些层的权重在 _iterate_weights 中被过滤掉了，
        需要通过扫描权重文件的 key 来确定哪些层和权重被排除了.

        Returns:
            包含被排除信息的字典:
            - "layers": 被排除的层号集合
            - "weight_names": 被排除的权重名称列表
        """
        if self.load_partial_layers is None:
            return {"layers": set(), "weight_names": []}

        model_path = self.model_path
        weights_files = glob(os.path.join(model_path, "*.safetensors"))

        if len(weights_files) == 0:
            return {"layers": set(), "weight_names": []}

        excluded_layers = set()
        excluded_weight_names = []

        # 快速扫描所有权重文件的 key，找出被排除的层和权重
        for st_file in weights_files:
            with safe_open(st_file, "pt", "cpu") as f:
                for name in f.keys():  # noqa: SIM118
                    if self._is_excluded_layer_weight(name):
                        layer_num = self._extract_layer_num(name)
                        if layer_num is not None:
                            excluded_layers.add(layer_num)
                        excluded_weight_names.append(name)

        return {"layers": excluded_layers, "weight_names": excluded_weight_names}
