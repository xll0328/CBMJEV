"""Persistent, hash-bound nested-OOF components (not deployment bundles)."""
from pathlib import Path

import torch

from .contracts import stable_hash
from .crossfit_training import _validate_head_binding
from .io import file_hash, fresh_dir, read_json, write_json
from .learning import MaskedHead, schema_signature


def save_crossfit_head(out, head, schema, report):
    """Save a frozen fold head without inventing a controller or changing identity."""
    _validate_head_binding(head, schema, report)
    out = fresh_dir(out)
    payload = {"format": "cbmjev-crossfit-head-v1",
               "schema_signature": schema_signature(schema),
               "config": head.config,
               "class_weights": head.class_weights,
               "state_dict": {k: v.detach().cpu() for k, v in head.network.state_dict().items()}}
    with (out / "head.pt").open("xb") as stream:
        torch.save(payload, stream)
    write_json(out / "head_report.json", report)
    receipt = {"format": "cbmjev-crossfit-head-artifact-v1",
               "checkpoint_sha256": file_hash(out / "head.pt"),
               "report_file_sha256": file_hash(out / "head_report.json"),
               "head_artifact_id": report["head_artifact_id"],
               "schema_signature": schema_signature(schema)}
    receipt["receipt_sha256"] = stable_hash(receipt)
    # Written last: a checkpoint alone is not a completed artifact.
    write_json(out / "receipt.json", receipt)
    return receipt


def load_crossfit_head(directory, schema, *, device="cpu"):
    """Restore on an execution device while preserving the hashed training config."""
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    core = dict(receipt)
    if core.pop("receipt_sha256", None) != stable_hash(core):
        raise ValueError("crossfit head receipt hash mismatch")
    if (receipt.get("format") != "cbmjev-crossfit-head-artifact-v1"
            or receipt.get("schema_signature") != schema_signature(schema)):
        raise ValueError("crossfit head artifact format/schema mismatch")
    for filename, key in (("head.pt", "checkpoint_sha256"),
                          ("head_report.json", "report_file_sha256")):
        if file_hash(directory / filename) != receipt.get(key):
            raise ValueError("crossfit head artifact bytes changed: " + filename)
    payload = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    if (payload.get("format") != "cbmjev-crossfit-head-v1"
            or payload.get("schema_signature") != schema_signature(schema)):
        raise ValueError("crossfit head checkpoint format/schema mismatch")
    identity_config = payload["config"]
    head = MaskedHead(schema, {**identity_config, "device": device},
                      class_weights=payload["class_weights"])
    head.network.load_state_dict(payload["state_dict"], strict=True)
    head.config = dict(identity_config)
    head.network.eval().requires_grad_(False)
    report = read_json(directory / "head_report.json")
    _validate_head_binding(head, schema, report)
    if report["head_artifact_id"] != receipt["head_artifact_id"]:
        raise ValueError("crossfit head artifact identity mismatch")
    return head, report
