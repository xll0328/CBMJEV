"""Versioned experiment configuration; fail on misspellings and unsafe budgets."""
import copy
import math

from .contracts import DeclaredCost


DEFAULT_CONFIG = {
    "schema_version": "cbmjev-config-v1", "seed": 17, "device": "cpu",
    "learning": {"objective": "risk", "hidden": 128, "head_epochs": 30,
                 "policy_epochs": 30, "batch_size": 64, "learning_rate": 0.001,
                 "masks_per_sample": 8, "include_pairs": True, "include_all": True,
                 "max_pair_actions": 8, "actions_per_state": 64,
                 "weight_decay": 0.0, "dropout": 0.0, "deterministic": True, "cpu_threads": 1,
                 "class_weighting": "none"},
    "policy": {"cost_weight": 0.03, "max_groups": None, "max_cost": None},
    "cost": {"setup": 0.0, "call": 0.0, "per_group": 1.0,
             "units": "declared_query_units"},
    "evaluation": {"methods": ["stop", "all", "fixed", "random", "static", "risk"],
                   "bootstrap_resamples": 1000, "lookahead_depth": 2},
}


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("unknown config keys: " + ", ".join(sorted(unknown)))
    result = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in config.items():
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError("config section must be an object: " + key)
            extras = set(value) - set(result[key])
            if key == "learning":
                extras -= {"pairs", "actions_per_state"}
            if extras:
                raise ValueError("unknown " + key + " options: " + ", ".join(sorted(extras)))
            result[key].update(value)
        else:
            result[key] = value
    if result["schema_version"] != "cbmjev-config-v1":
        raise ValueError("expected cbmjev-config-v1")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(result["device"], str) or not result["device"]:
        raise ValueError("device must be explicit")
    DeclaredCost(**result["cost"])
    for key in ("cost_weight", "max_cost"):
        value = result["policy"][key]
        if value is None and key == "max_cost":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
    limit = result["policy"]["max_groups"]
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("max_groups must be a nonnegative integer or null")
    objective = result["learning"]["objective"]
    if objective not in ("risk", "value"):
        raise ValueError("objective must be risk or value")
    if result["learning"]["class_weighting"] not in ("none", "inverse_frequency"):
        raise ValueError("class_weighting must be none or inverse_frequency")
    allowed = {"stop", "all", "fixed", "random", "static", "risk", "value", "static_value", "value_singleton", "lookahead", "nano_risk", "nano_static_risk", "matched_mlp_risk", "brig", "learned_static_mask"}
    methods = result["evaluation"]["methods"]
    if not isinstance(methods, list) or not methods or any(m not in allowed for m in methods):
        raise ValueError("invalid evaluation methods")
    if len(set(methods)) != len(methods):
        raise ValueError("evaluation methods must be unique")
    if objective == "value" and "risk" in methods and "methods" not in config.get("evaluation", {}):
        result["evaluation"]["methods"] = ["value" if m == "risk" else m for m in methods]
    methods = result["evaluation"]["methods"]
    if (objective == "value" and "risk" in methods) or (objective == "risk" and set(methods) & {"value", "static_value", "value_singleton"}):
        raise ValueError("risk/value evaluation must use the matching trained objective")
    if type(result["evaluation"]["bootstrap_resamples"]) is not int or result["evaluation"]["bootstrap_resamples"] < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    return result


def learning_config(config):
    return {**config["learning"], "seed": config["seed"], "device": config["device"]}
