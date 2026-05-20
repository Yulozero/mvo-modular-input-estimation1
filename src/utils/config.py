import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config_from_preset(base_path: str, preset_path: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_cfg = load_yaml(base_path)
    preset_cfg = load_yaml(preset_path)
    cfg = deep_update(base_cfg, preset_cfg)
    if override:
        cfg = deep_update(cfg, override)

    cfg["_meta"] = {
        "base_path": base_path,
        "preset_path": preset_path,
        "preset_name": preset_cfg.get("preset_name"),
        "description": preset_cfg.get("description", ""),
    }
    return cfg


def load_experiment_config(exp_path: str) -> Dict[str, Any]:
    exp_cfg = load_yaml(exp_path)
    base_path = exp_cfg.get("base")
    preset_path = exp_cfg.get("preset")
    override = exp_cfg.get("override", {})

    if not base_path:
        raise ValueError("Experiment config must contain 'base'")
    if not preset_path:
        raise ValueError("Experiment config must contain 'preset'")

    cfg = load_config_from_preset(base_path, preset_path, override=override)
    cfg["_meta"]["experiment_path"] = exp_path
    cfg["_meta"]["experiment_name"] = exp_cfg.get("experiment_name", override.get("output", {}).get("experiment_name"))
    return cfg


def save_json(data: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
