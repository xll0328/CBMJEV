"""Train an objective-matched structured Choice utility-regression control.

Requires a completed soft-CE Choice pair from the same source packages. This
standalone entry leaves the original source-bound implementation untouched.
It is validation-development evidence, not test evaluation or full JEV.
"""

import argparse
from collections import Counter
from pathlib import Path
import random

import torch

from cbmjev.choice_artifacts import (_inputs, _preflight, _read_heads, _reservoir,
                                     _source_binding, load_choice_pair)
from cbmjev.choice_features import (CHOICE_FEATURE_VERSION, choice_feature_width,
                                    encode_choice_features)
from cbmjev.choice_risk_training import fit_choice_utility_pair
from cbmjev.choice_targets import singleton_choice_example
from cbmjev.choice_training import _source_hash, _tensor_hash
from cbmjev.contracts import stable_hash
from cbmjev.decision_sets import iter_decision_sets
from cbmjev.io import file_hash, fresh_dir, read_json, write_json


MATCHED_KEYS = ("source_content_sha256", "effective_features_sha256",
    "derived_ids_in_order", "initial_shared_tensors_sha256",
    "initial_active_logits_sha256", "epoch_order_sha256", "order_sha256",
    "questions_exposed_per_head", "candidates_exposed_per_head",
    "optimizer_steps_per_head")


def _binding(prepared, targets, config, soft_choice):
    module_root = Path(__file__).resolve().parents[1] / "cbmjev"
    return {"inputs": _inputs(prepared, targets, config),
            "original_choice_source": _source_binding(),
            "risk_control_source": {
                "choice_risk_loss.py": file_hash(module_root / "choice_risk_loss.py"),
                "choice_risk_training.py": file_hash(module_root / "choice_risk_training.py"),
                "train_choice_risk_pair.py": file_hash(Path(__file__))},
            "soft_choice_receipt_sha256": file_hash(Path(soft_choice) / "receipt.json")}


def _verified_soft_metadata(soft_choice, binding):
    """Read receipt-bound soft metadata without a second full target scan.

    The later ``_preflight`` still validates every source package and lineage.
    This helper does not load model tensors because only their receipt-bound
    initial-weight hashes are needed for the matched training control.
    """
    soft_choice = Path(soft_choice)
    receipt = read_json(soft_choice / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_sha256", None) != stable_hash(unsigned)
            or receipt.get("format") != "cbmjev-structured-choice-pair-artifact-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("bindings") != {
                "inputs": binding["inputs"],
                "source": binding["original_choice_source"]}):
        raise ValueError("soft Choice receipt/source binding mismatch")
    for name in ("scalar.pt", "attention.pt", "config.json", "report.json"):
        if receipt["files"].get(name) != file_hash(soft_choice / name):
            raise ValueError("soft Choice artifact checksum mismatch: " + name)
    settings, report = read_json(soft_choice / "config.json"), read_json(soft_choice / "report.json")
    if (settings.get("format") != "cbmjev-structured-choice-pair-config-v1"
            or report.get("format") != "cbmjev-structured-choice-pair-report-v1"
            or settings.get("feature_version") != CHOICE_FEATURE_VERSION
            or settings.get("schema_hash") != receipt.get("schema_hash")
            or report.get("package_bindings") != receipt.get("package_bindings")):
        raise ValueError("soft Choice config/report binding mismatch")
    _read_heads(soft_choice, settings, report, "cpu")
    return settings, report, receipt


def _selected(prepared, targets, config, settings):
    schema, cfg, packages = _preflight(prepared, targets, config,
                                       validate_decisions=False)
    if settings["schema_hash"] != schema.hash:
        raise ValueError("soft Choice schema mismatch")
    seed = settings["seed"]
    budget_mode = settings["budget_mode"]
    budget = settings["remaining_groups"]

    def stream():
        budget_rng = random.Random("choice-budget-v1:" + str(seed))
        for package in packages:
            for decision in iter_decision_sets(package, schema, cfg):
                available = schema.num_groups - sum(schema.group_mask(decision.model_inputs.observed))
                current_budget = (budget_rng.randrange(available + 1)
                    if budget_mode == "uniform_remaining" else budget)
                yield singleton_choice_example(decision, schema,
                    remaining_groups=current_budget,
                    cost_weight=settings["cost_weight"],
                    temperature=settings["temperature"])

    selected, count = _reservoir(stream(), settings["max_questions"], seed)
    features = tuple(encode_choice_features(e.model_inputs, schema) for e in selected)
    if choice_feature_width(schema) != settings["feature_width"]:
        raise ValueError("structured feature width mismatch")
    bindings = [{"path": str(Path(path).resolve()),
                 "package_sha256": package["package_sha256"],
                 "target_artifact_id": package["target_artifact_id"]}
                for path, package in zip(targets, packages)]
    return schema, selected, features, count, bindings


def train_choice_risk_pair(prepared, targets, config, soft_choice, out):
    prepared, config, soft_choice, out = map(Path, (prepared, config, soft_choice, out))
    targets = tuple(map(Path, targets))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("risk-control output must be new or empty")
    before = _binding(prepared, targets, config, soft_choice)
    source_settings, source_report, soft_receipt = _verified_soft_metadata(soft_choice, before)
    if source_settings.get("feature_backend") != "structured":
        raise ValueError("risk control requires the structured soft-Choice arm")
    if source_report["fit"]["config"]["loss"] != "question-mean-soft-CE":
        raise ValueError("source is not the declared soft-CE objective")
    schema, selected, features, count, package_bindings = _selected(
        prepared, targets, config, source_settings)
    soft_fit = source_report["fit"]
    if (package_bindings != soft_receipt["package_bindings"]
            or count != source_report["available_occurrences"]
            or tuple(e.derived_id for e in selected) != tuple(source_report["selected_derived_ids"])
            or _source_hash(selected, features) != soft_fit["source_content_sha256"]
            or stable_hash([_tensor_hash(v) for v in features]) != soft_fit["effective_features_sha256"]):
        raise ValueError("risk control does not reproduce soft-Choice training examples/features")
    if before != _binding(prepared, targets, config, soft_choice):
        raise ValueError("risk-control source changed during preflight")
    risk_args = {"epochs": source_settings["epochs"], "batch_size": source_settings["batch_size"],
                 "lr": source_settings["learning_rate"], "seed": source_settings["seed"],
                 "device": source_settings["device"]}
    scalar, attention, fit = fit_choice_utility_pair(selected, features, **risk_args)
    for key in MATCHED_KEYS:
        if fit[key] != soft_fit[key]:
            raise ValueError("objective control mismatch: " + key)
    for name in ("scalar", "attention"):
        if fit["heads"][name]["initial_weights_sha256"] != soft_fit["heads"][name]["initial_weights_sha256"]:
            raise ValueError("objective control initial weights differ: " + name)
    if before != _binding(prepared, targets, config, soft_choice):
        raise ValueError("risk-control source changed during training")

    settings = {**source_settings,
        "format": "cbmjev-structured-choice-risk-control-config-v1",
        "objective": "question-mean-active-utility-MSE",
        "soft_choice_receipt_sha256": before["soft_choice_receipt_sha256"],
        "paper_evidence": False}
    report = {"format": "cbmjev-structured-choice-risk-control-report-v1",
        "evidence_status": "DEVELOPMENT_ONLY", "source_binding": before,
        "matched_soft_choice_fit_sha256": stable_hash(soft_fit),
        "matched_keys": list(MATCHED_KEYS),
        "available_occurrences": count, "selected_occurrences": len(selected),
        "selected_budget_counts": dict(sorted(Counter(str(e.model_inputs.remaining_groups)
                                                    for e in selected).items())),
        "selected_derived_ids": [e.derived_id for e in selected],
        "interpretation": "one-step utility regression, not Bellman Q or a JEV-specific method",
        "fit": fit}
    destination = fresh_dir(out)
    for name, head in (("scalar", scalar), ("attention", attention)):
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()},
                   destination / (name + ".pt"))
    write_json(destination / "config.json", settings)
    write_json(destination / "report.json", report)
    _read_heads(destination, settings, report, "cpu")
    if before != _binding(prepared, targets, config, soft_choice):
        raise ValueError("risk-control source changed during export")
    receipt = {"format": "cbmjev-structured-choice-risk-control-artifact-v1",
        "status": "COMPLETE", "schema_hash": schema.hash, "bindings": before,
        "files": {name: file_hash(destination / name)
                  for name in ("scalar.pt", "attention.pt", "config.json", "report.json")}}
    receipt["receipt_sha256"] = stable_hash(receipt)
    write_json(destination / "receipt.json", receipt)
    return receipt


def load_choice_risk_pair(directory, *, prepared, targets, config, soft_choice,
                          device="cpu"):
    directory, targets = Path(directory), tuple(map(Path, targets))
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_sha256", None) != stable_hash(unsigned)
            or receipt.get("format") != "cbmjev-structured-choice-risk-control-artifact-v1"
            or receipt.get("status") != "COMPLETE"):
        raise ValueError("invalid risk-control receipt")
    for name in ("scalar.pt", "attention.pt", "config.json", "report.json"):
        if receipt["files"].get(name) != file_hash(directory / name):
            raise ValueError("risk-control artifact checksum mismatch")
    _, _, soft_settings, soft_report = load_choice_pair(soft_choice,
        prepared=prepared, targets=targets, config=config, device="cpu")
    if receipt["bindings"] != _binding(prepared, targets, config, soft_choice):
        raise ValueError("risk-control source/input binding mismatch")
    settings, report = read_json(directory / "config.json"), read_json(directory / "report.json")
    if (settings["schema_hash"] != receipt["schema_hash"]
            or settings["soft_choice_receipt_sha256"] != receipt["bindings"]["soft_choice_receipt_sha256"]
            or report["source_binding"] != receipt["bindings"]
            or report["matched_soft_choice_fit_sha256"] != stable_hash(soft_report["fit"]) or
            any(settings[key] != soft_settings[key] for key in
                ("schema_hash", "feature_version", "feature_width", "feature_backend",
                 "max_questions", "seed", "remaining_groups", "budget_mode",
                 "cost_weight", "temperature", "epochs", "batch_size",
                 "learning_rate", "device"))):
        raise ValueError("risk-control settings/source mismatch")
    heads = _read_heads(directory, settings, report, device)
    if receipt["bindings"] != _binding(prepared, targets, config, soft_choice):
        raise ValueError("risk-control sources changed during reload")
    return *heads, settings, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "soft-choice", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args()
    receipt = train_choice_risk_pair(args.prepared, args.targets, args.config,
                                     args.soft_choice, args.out)
    print(receipt["receipt_sha256"])


if __name__ == "__main__":
    main()
