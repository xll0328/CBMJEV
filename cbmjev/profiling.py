"""Validation-only live responder profiling, never cached-replay latency."""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

from .contracts import stable_hash
from .io import environment, file_hash, fresh_dir, write_json
from .runtime import LiveEnvironment, synchronize

HASH_SCOPE = "model_facing_UTF8_text_and_EXIF_clean_PNG_bytes_not_raw_file_paths"
TIMING_SCOPE = "loaded_model_new_episode_session_serial_backend_including_lazy_preprocessing"


def _plans(schema, batch_sizes):
    sizes = tuple(batch_sizes)
    if any(type(size) is not int or size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("batch_sizes must contain unique positive integers")
    groups = tuple(range(schema.num_groups))
    result = {"all": (groups,)}
    for size in sizes:
        result[f"batch_{size}"] = tuple(groups[start:start+size] for start in range(0, len(groups), size))
    return result


def _validate_row(row):
    if row.get("split") != "validation":
        raise ValueError("profiling accepts validation rows only; no test/fit fallback")
    if any(not isinstance(row.get(key), str) or not row[key] for key in ("sample_id", "group_id")):
        raise ValueError("profile row needs nonempty sample_id and group_id")
    if not isinstance(row.get("input"), dict):
        raise ValueError("profile row needs canonical input mapping")


def _one_run(schema, row, responder, actions, name, repeat, raw_root, device, model_hash, warmup=False):
    synchronize(device)
    start = time.perf_counter()
    env = LiveEnvironment(row["input"], schema, responder, raw_root=raw_root, device=device)
    measurements = []
    for index, action in enumerate(actions):
        before_response, before_preprocess = env.responder_ms, env.preprocess_ms
        begin = time.perf_counter()
        values = env.query(action)
        duration = (time.perf_counter()-begin)*1000
        stats = env.backend_stats[-1]
        response_ms = env.responder_ms-before_response
        measurements.append({"call_index": index, "groups": list(action), "atom_ids": list(schema.expand(action)),
            "values": list(values), "query_wall_ms": duration, "response_ms": response_ms,
            "preprocess_ms": env.preprocess_ms-before_preprocess, "backend_stats": stats})
    synchronize(device)
    total_ms = (time.perf_counter()-start)*1000
    # External instrumentation reads sanitized content only AFTER the measured episode.
    # It does not send timing/hash/IDs to responder or controller and never decodes again.
    payload = env._payload
    begin = time.perf_counter()
    content = {"text": payload.text, "images": [hashlib.sha256(image).hexdigest() for image in payload.images]}
    digest = stable_hash(content)
    digest_ms = (time.perf_counter()-begin)*1000
    first = measurements[0]
    shared = first["backend_stats"].get("cost_mode") == "SHARED_CHEAP_ALL_HEADS"
    return {"sample_id": row["sample_id"], "group_id": row["group_id"], "split": "validation",
        "configuration": name, "batch_size_requested": None if name == "all" else int(name.split("_")[1]),
        "max_groups_per_call": max(map(len, actions)), "repeat": repeat, "is_warmup": warmup,
        "groups": list(range(schema.num_groups)), "calls": env.calls, "schema_hash": schema.hash,
        "model_hash": model_hash, "input_digest": digest, "bytehash_scope": HASH_SCOPE,
        "input_digest_ms_excluded": digest_ms, "total_wall_ms": total_ms,
        "preprocess_ms": env.preprocess_ms, "response_ms": env.responder_ms,
        "coordinator_ms": max(0., total_ms-env.preprocess_ms-env.responder_ms),
        "first_call_response_ms": first["response_ms"],
        "shared_encoder_first_call_ms": first["response_ms"] if shared else None,
        "shared_encoder_first_call_scope": "entire_shared_encoder_plus_all_heads_response_not_isolated_encoder" if shared else None,
        "final_values": list(env.state()), "measurements": measurements,
        "timing_scope": TIMING_SCOPE, "cold_session": True, "cold_model": False,
        "evidence_status": "LIVE_VALIDATION_PROFILE_NOT_PAPER_SPEEDUP"}


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low]*(high-position) + ordered[high]*(position-low) if high != low else ordered[low]


def _stats(values):
    return {"n": len(values), "mean": sum(values)/len(values), "p50": _percentile(values, .5),
            "p95": _percentile(values, .95), "min": min(values), "max": max(values)}


def _emit(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
    handle.flush()


def profile_inputs(schema, input_rows, responder, out, *, raw_root=None, device="cpu", limit=100,
                   repeats=3, batch_sizes=(1, 2, 4), warmup=3, model_hash="INJECTED_UNVERIFIED", provenance=None):
    """Stream canonical validation rows. Warmup uses first selected case; no fitting."""
    for name, value in (("limit", limit), ("repeats", repeats), ("warmup", warmup)):
        if type(value) is not int or value < (0 if name == "warmup" else 1):
            raise ValueError(f"invalid {name}")
    plans = _plans(schema, batch_sizes)
    iterator = iter(input_rows)
    first = next(iterator, None)
    if first is None:
        raise ValueError("no validation rows to profile")
    _validate_row(first)
    out = fresh_dir(out)
    metrics = ("total_wall_ms", "preprocess_ms", "response_ms", "coordinator_ms", "first_call_response_ms")
    aggregates = {name: {metric: [] for metric in metrics} for name in plans}
    seen_ids, seen_groups = set(), set()
    consistency, repeated_full_stable = True, True
    comparisons, mismatches, full_repeat_comparisons, full_repeat_mismatches = 0, 0, 0, 0
    measured_count = 0
    with (out / "warmup.jsonl").open("x", encoding="utf-8") as warm_file:
        for repeat in range(warmup):
            for name, actions in plans.items():
                _emit(warm_file, _one_run(schema, first, responder, actions, name, repeat,
                    raw_root, device, model_hash, warmup=True))
    with (out / "timings.jsonl").open("x", encoding="utf-8") as timings:
        for case_index, row in enumerate(itertools.islice(itertools.chain([first], iterator), limit)):
            _validate_row(row)
            if row["sample_id"] in seen_ids:
                raise ValueError("duplicate sample_id in validation profiling rows")
            seen_ids.add(row["sample_id"])
            seen_groups.add(row["group_id"])
            first_full, first_digest = None, None
            for repeat in range(repeats):
                # Deterministic rotation reduces a fixed always-first configuration bias.
                names = list(plans)
                offset = (case_index+repeat) % len(names)
                records = {}
                for name in names[offset:]+names[:offset]:
                    records[name] = _one_run(schema, row, responder, plans[name], name, repeat,
                                            raw_root, device, model_hash)
                reference = records["all"]["final_values"]
                current_digest = records["all"]["input_digest"]
                if first_full is None:
                    first_full, first_digest = reference, current_digest
                else:
                    full_repeat_comparisons += 1
                    stable = reference == first_full
                    repeated_full_stable &= stable
                    full_repeat_mismatches += not stable
                for name, record in records.items():
                    if record["input_digest"] != first_digest:
                        raise ValueError("input bytes changed between profile configurations/repeats")
                    same = record["final_values"] == reference
                    record["matches_all_same_repeat"] = same
                    record["matches_all_first_repeat"] = record["final_values"] == first_full
                    if name != "all":
                        comparisons += 1
                        mismatches += not same
                        consistency &= same
                    _emit(timings, record)
                    measured_count += 1
                    for metric in metrics:
                        aggregates[name][metric].append(record[metric])
    summary = {"format": "cbmjev-live-validation-profile-v1", "split": "validation", "n_cases": len(seen_ids),
        "n_groups": len(seen_groups), "repeats": repeats, "limit": limit, "device": str(device),
        "warmup_episodes": warmup*len(plans), "measured_episodes": measured_count,
        "schema_hash": schema.hash, "model_hash": model_hash, "bytehash_scope": HASH_SCOPE,
        "timing_scope": TIMING_SCOPE, "cold_session": True, "cold_model": False,
        "model_loading_timed": False, "model_warmup_episodes_per_configuration": warmup,
        "task_head_or_controller_timed": False, "concurrency": 1,
        "selection": "first_limit_validation_rows_in_frozen_input_order",
        "statistics_unit": "episode; repeated timings are not independent clinical cases",
        "cost_units": "measured_milliseconds_not_a_fitted_declared_cost",
        "configurations": {name: {"actions": [list(a) for a in plans[name]],
            "timings": {metric: _stats(values) for metric, values in values_by_metric.items()}}
            for name, values_by_metric in aggregates.items()},
        "batch_independent": (False if not consistency or not repeated_full_stable else
                              True if comparisons > 0 else None),
        "batch_independence_scope": "empirical_hard_output_equality_only_on_profiled_cases_configs_repeats",
        "full_vs_split_comparisons": comparisons, "full_vs_split_mismatches": mismatches,
        "full_repeat_comparisons": full_repeat_comparisons, "full_repeat_mismatches": full_repeat_mismatches,
        "flat_cache_certified": False, "training_configuration_modified": False,
        "provenance": provenance or {}, "raw_timings_sha256": file_hash(out / "timings.jsonl"),
        "evidence_status": "LIVE_VALIDATION_PROFILE_NOT_PAPER_SPEEDUP"}
    write_json(out / "summary.json", summary)
    write_json(out / "environment.json", environment())
    return {"out": str(out), **summary}


def profile_backend(prepared, responder_dir, out, raw_root=None, device="cpu", limit=100,
                    repeats=3, batch_sizes=(1, 2, 4), warmup=3):
    # Lazy import: pipeline may expose a CLI wrapper around this module.
    from .pipeline import load_prepared, load_backend
    schema, rows, membership = load_prepared(prepared)
    responder, receipt = load_backend(responder_dir, schema, device)
    membership_hash = file_hash(Path(prepared) / "membership.jsonl")
    if receipt.get("membership_sha256") != membership_hash:
        raise ValueError("responder fit role manifest differs from prepared profiling data")
    protected = set(receipt.get("supervised_group_ids", ()))

    def validation_rows():
        for row in rows:
            role = membership[row["sample_id"]]["split"]
            if role != "validation":
                continue
            if row["group_id"] in protected:
                raise ValueError("validation profile overlaps responder supervised groups")
            yield {"sample_id": row["sample_id"], "group_id": row["group_id"],
                   "split": role, "input": row["input"]}

    return profile_inputs(schema, validation_rows(), responder, out, raw_root=raw_root, device=device,
        limit=limit, repeats=repeats, batch_sizes=batch_sizes, warmup=warmup,
        model_hash=receipt["checkpoint_sha256"], provenance={
            "responder_artifact_id": receipt.get("artifact_id"), "responder_kind": receipt["kind"],
            "responder_receipt_sha256": file_hash(Path(responder_dir) / "receipt.json"),
            "prepared_samples_sha256": file_hash(Path(prepared) / "samples.jsonl"),
            "membership_sha256": membership_hash})
