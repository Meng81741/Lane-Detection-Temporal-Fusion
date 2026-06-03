"""Configuration loading and merging utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Supports inheritance via `_base_: path/to/base.yaml` key.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = _deep_merge(base_config, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base."""
    return _deep_merge(base, override)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def config_to_namespace(config: Dict[str, Any]):
    """Convert a nested dict to a namespace object for dot-access."""
    class Namespace:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, Namespace(**v))
                else:
                    setattr(self, k, v)

        def __repr__(self):
            items = [f"{k}={v}" for k, v in self.__dict__.items()]
            return f"Config({', '.join(items)})"

    return Namespace(**config)
