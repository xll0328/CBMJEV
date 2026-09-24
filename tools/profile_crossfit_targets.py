#!/usr/bin/env python3
"""CPU-only, synthetic CUB-shaped target-storage profiling; never paper evidence."""
import argparse
from pathlib import Path
import random
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbmjev.contracts import Concept, QueryGroup, Schema, stable_hash
from cbmjev.crossfit_artifacts import save_crossfit_head, load_crossfit_head
from cbmjev.crossfit_targets import open_target_package
from cbmjev.crossfit_training import (fit_head_only, construct_action_targets,
                                     _validate_target_package)
from cbmjev.io import environment, file_hash, fresh_dir, write_json
from cbmjev.learning import normalize_config
from cbmjev.pipeline import code_fingerprint
from cbmjev.provenance import make_fit_record


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def profile(out, *, rows=128, epochs=1, head_rows=200, head_epochs=1,
            seed=60, hidden=256, batch_size=128):
    for name, value in (("rows", rows), ("epochs", epochs), ("head_rows", head_rows),
                        ("head_epochs", head_epochs), ("hidden", hidden),
                        ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    out = fresh_dir(out)
    source_hash = code_fingerprint()
    tool_hash = file_hash(__file__)
    phases = []

    def measure(name, operation):
        before = _rss()
        start = time.perf_counter()
        result = operation()
        phases.append({"phase": name, "elapsed_seconds": time.perf_counter() - start,
                       "process_high_water_rss_before_bytes": before,
                       "process_high_water_rss_after_bytes": _rss()})
        return result

    schema = Schema("SYNTHETIC_CUB_SHAPE_NOT_CUB_DATA", 200,
        tuple(Concept("atom_%03d" % i, "synthetic binary atom %d" % i, ("no", "yes"))
              for i in range(312)),
        tuple(QueryGroup("group_%02d" % g, tuple(range(g * 312 // 28, (g + 1) * 312 // 28)))
              for g in range(28)))
    cfg = normalize_config({"seed": seed, "device": "cpu", "hidden": hidden,
        "head_epochs": head_epochs, "policy_epochs": epochs, "batch_size": batch_size,
        "masks_per_sample": 4, "include_pairs": True, "include_all": True,
        "max_pair_actions": 8, "actions_per_state": 64, "objective": "risk"}, schema)
    fixture_config = {"rows": rows, "head_rows": head_rows, "seed": seed,
                      "generator": "independent_random_binary_z_uniform_y_v1"}

    def fixture():
        rng = random.Random(seed)
        def generate(prefix, count):
            return [{"sample_id": "%s-%06d" % (prefix, i),
                     "group_id": "%s-%06d" % (prefix, i),
                     "z": [rng.randrange(2) for _ in range(312)],
                     "y": rng.randrange(200)} for i in range(count)]
        return generate("head", head_rows), generate("target", rows)

    head_data, target_data = measure("synthetic_fixture_generation", fixture)
    # This is an explicitly synthetic generator, not a claim of a fitted visual
    # responder. Its fixed outputs and no-supervision ancestry are hash-bound.
    source_id = "synthetic-generator:" + stable_hash({"config": fixture_config,
                              "head_rows": head_data, "target_rows": target_data})
    provenance = [make_fit_record(source_id, supervised_group_ids=[],
        fit_kind="frozen_external", metadata={"synthetic_fixture": True,
        "generator_config": fixture_config, "not_a_trained_responder": True})]
    def assignments(records):
        return {row["group_id"]: source_id for row in records}

    head, report = measure("actual_masked_head_fit", lambda: fit_head_only(
        head_data, schema, cfg, response_artifact_by_group=assignments(head_data),
        provenance_records=provenance))
    save_crossfit_head(out / "head", head, schema, report)
    head, report = measure("persisted_head_full_verification",
                          lambda: load_crossfit_head(out / "head", schema, device="cpu"))
    manifest = out / "action_targets.json"
    package = measure("construct_write_hash_and_full_prepublication_validation",
        lambda: construct_action_targets(target_data, head, schema, cfg,
            head_report=report, response_artifact_by_group=assignments(target_data),
            provenance_records=provenance, manifest_path=manifest))
    loaded = measure("open_storage_manifest", lambda: open_target_package(manifest))
    count = measure("read_all_records_with_shard_integrity",
                    lambda: sum(1 for _ in loaded["records"]))
    if count != package["record_count"]:
        raise AssertionError("full record scan count differs")
    measure("full_logical_package_validation",
            lambda: _validate_target_package(loaded, schema, cfg))
    if code_fingerprint() != source_hash or file_hash(__file__) != tool_hash:
        raise ValueError("source changed during profile")
    target_files = [manifest] + sorted((out / "action_targets_records").glob("*.jsonl"))
    result = {"format": "cbmjev-crossfit-target-profile-v1", "status": "COMPLETE",
        "evidence_status": "SYNTHETIC_CPU_STORAGE_PROFILE_NOT_RESEARCH_RESULT",
        "config": cfg, "fixture": fixture_config, "schema_hash": schema.hash,
        "num_atoms": 312, "num_groups": 28, "num_classes": 200,
        "source_code_hash": source_hash, "tool_sha256": tool_hash,
        "environment": environment(), "phases": phases, "record_count": count,
        "records_per_epoch": package["records_per_epoch"],
        "package_sha256": package["package_sha256"],
        "target_artifact_id": package["target_artifact_id"],
        "target_storage_bytes": sum(path.stat().st_size for path in target_files),
        "target_files_sha256": {str(p.relative_to(out)): file_hash(p) for p in target_files},
        "full_record_scan_completed": True, "full_logical_validation_completed": True,
        "rss_scope": "cumulative process high-water RSS; not per-phase allocation or GPU memory",
        "limitations": ["No real CUB data or trained visual responder.",
                        "Warm filesystem cache may affect reread times.",
                        "No extrapolated wall-clock promise or main-experiment claim."]}
    write_json(out / "profile.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    for option, default in (("rows", 128), ("epochs", 1), ("head-rows", 200),
                            ("head-epochs", 1), ("seed", 60), ("hidden", 256),
                            ("batch-size", 128)):
        parser.add_argument("--" + option, type=int, default=default)
    result = profile(**vars(parser.parse_args(argv)))
    print("COMPLETE: %d fully validated synthetic records; %d storage bytes" %
          (result["record_count"], result["target_storage_bytes"]))


if __name__ == "__main__":
    main()
