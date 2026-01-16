import importlib
import os
import pkgutil

_MODELS_MAPPING = {}


def register_model(name):
    def decorator(cls):
        _MODELS_MAPPING[name] = cls
        return cls

    return decorator


def _discover_models():
    # 自动导入当前目录下所有的 py 文件以触发注册
    models_dir = os.path.dirname(__file__)
    for _, name, is_pkg in pkgutil.iter_modules([models_dir]):
        if name != "models_mapping":
            importlib.import_module(f"nanovllm.models.{name}")


class ModelsMappingDict(dict):
    def __getitem__(self, key):
        if not _MODELS_MAPPING:
            _discover_models()
        return _MODELS_MAPPING[key]

    def __contains__(self, key):
        if not _MODELS_MAPPING:
            _discover_models()
        return key in _MODELS_MAPPING

    def get(self, key, default=None):
        if not _MODELS_MAPPING:
            _discover_models()
        return _MODELS_MAPPING.get(key, default)

    def __iter__(self):
        if not _MODELS_MAPPING:
            _discover_models()
        return iter(_MODELS_MAPPING)

    def keys(self):
        if not _MODELS_MAPPING:
            _discover_models()
        return _MODELS_MAPPING.keys()


MODELS_MAPPING = ModelsMappingDict()
