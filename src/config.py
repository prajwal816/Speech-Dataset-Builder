"""
Configuration loader for the Speech AI Dataset Builder.

Loads YAML configs with dot-access support and sensible defaults.
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional


class Config:
    """Hierarchical config with dot-access backed by a YAML file."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely retrieve a nested key using dot notation (e.g. 'audio.target_sample_rate')."""
        keys = key.split(".")
        obj = self
        for k in keys:
            if isinstance(obj, Config):
                obj = getattr(obj, k, None)
            elif isinstance(obj, dict):
                obj = obj.get(k)
            else:
                return default
            if obj is None:
                return default
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to a plain dict."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: str) -> Config:
    """
    Load a YAML configuration file and return a ``Config`` object.

    Parameters
    ----------
    path : str
        Absolute or relative path to a YAML config file.

    Returns
    -------
    Config
        Parsed configuration with dot-access attributes.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    logging.getLogger("config").info("Loaded config from %s", path)
    return Config(data or {})
