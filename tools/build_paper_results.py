#!/usr/bin/env python3
"""Build development-only LaTeX/JSON from existing validation artifacts (stdlib only).

Examples:
  python3 tools/build_paper_results.py \
    --summary cebab/cohort=summary.json --out NEW_DIRECTORY
  python3 tools/build_paper_results.py \
    --audit-run hf_v1=runs/seed40 --audit-run hf_v1=runs/seed41 --out NEW_DIRECTORY

No selection of a best operating point, statistical test, claim promotion,
experiment, or source-file mutation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


SWEEP_FORMAT = "cbmjev-validation-cost-sweep-v1"
AUDIT_FORMAT = "cbmjev-semantic-response-audit-v1"
SWEEP_METRICS = ("accuracy", "macro_f1", "group_mean_risk", "mean_queried_groups",
                 "mean_calls", "mean_declared_cost")
AUDIT_METRICS = ("macro_concept_accuracy", "macro_concept_f1")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _finite(value, name, *, unit=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, not a boolean/string")
    if not math.isfinite(value) or value < 0 or (unit and value > 1):
        raise ValueError(f"{name} outside finite {'[0,1]' if unit else 'nonnegative'} range")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be nonempty printable text")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return value


def _false_flags(obj, *, require=None):
    if require and obj.get(require) is not False:
        raise ValueError(f"{require} must explicitly equal false")
    for key in ("paper_claim", "paper_evidence", "test_evaluated"):
        if key in obj and obj[key] is not False:
            raise ValueError(f"{key} must equal false")
    if "split" in obj and obj["split"] != "validation":
        raise ValueError("only split=validation is accepted")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(document, dict):
        raise ValueError(f"JSON object required: {path}")
    return document, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _spec(spec, *, summary):
    if "=" not in spec:
        raise ValueError("expected DATASET/COHORT=PATH or COHORT=RUN_DIR")
    name, path = spec.split("=", 1)
    labels = name.split("/")
    if len(labels) != (2 if summary else 1) or not all(LABEL.fullmatch(x) for x in labels) or not path:
        raise ValueError("use safe nonempty DATASET/COHORT labels (letters, digits, ._-)")
    return labels, Path(path)


def _seed_claim(obj, actual):
    if "num_seeds" in obj and _integer(obj["num_seeds"], "num_seeds", 1) != len(actual):
        raise ValueError("declared num_seeds does not match distinct point seeds")
    if "seeds" in obj:
        seeds = obj["seeds"]
        if not isinstance(seeds, list):
            raise ValueError("seeds must be a list")
        checked = [_integer(s, "seed") for s in seeds]
        if len(checked) != len(set(checked)) or sorted(checked) != sorted(actual):
            raise ValueError("declared seeds do not match point seeds")


def _stats(rows, names):
    return {name: {"mean": statistics.mean(r[name] for r in rows),
                   "sd": statistics.stdev(r[name] for r in rows) if len(rows) > 1 else None}
            for name in names}


def load_summary(spec):
    (dataset, cohort), path = _spec(spec, summary=True)
    doc, provenance = _read(path)
    if doc.get("format") != SWEEP_FORMAT:
        raise ValueError(f"unsupported summary format: {path}")
    _false_flags(doc, require="paper_claim")
    if "dataset" in doc and doc["dataset"] != dataset:
        raise ValueError("summary dataset conflicts with explicit input label")
    points = doc.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("nonempty points required; aggregates alone are not seed evidence")
    rows, groups = [], defaultdict(list)
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError("each point must be an object")
        _false_flags(point)
        if "dataset" in point and point["dataset"] != dataset:
            raise ValueError("point dataset mismatch")
        context = {}
        for field in ("schema_hash", "source_revision", "backend"):
            if field in point and field in doc and point[field] != doc[field]:
                raise ValueError(f"point/top-level {field} mismatch")
            context[field] = point.get(field, doc.get(field))
        for evidence_source in (doc, point):
            if str(evidence_source.get("source_revision", "")).upper().startswith("SYNTHETIC") or "SYNTHETIC" in str(evidence_source.get("data_evidence", "")).upper():
                raise ValueError("synthetic summary is not research exploratory evidence")
        mode = point.get("mode")
        expected = {"offline_replay": "OFFLINE_REPLAY_NOT_LATENCY",
                    "live": "LIVE_SINGLE_RUN_REQUIRES_REPLICATION"}
        if mode not in expected or point.get("evidence_status") != expected[mode]:
            raise ValueError("unknown/mismatched mode and evidence_status")
        row = {"kind": "cost_sweep", "dataset": dataset, "cohort": cohort, "split": "validation",
               "mode": mode, "paper_claim": False, "seed": _integer(point.get("seed"), "seed"),
               "method": _text(point.get("method"), "method"),
               "policy_id": _text(point.get("policy_id"), "policy_id"),
               "cost_weight": _finite(point.get("cost_weight"), "cost_weight"),
               "max_groups": _integer(point.get("max_groups"), "max_groups"),
               "num_samples": _integer(point.get("num_samples"), "num_samples", 1),
               "num_groups": _integer(point.get("num_groups"), "num_groups", 1),
               "evidence_status": point["evidence_status"],
               "source": provenance, "source_row": index,
               "cohort_context": context,
               "run": _text(point.get("run"), "run")}
        for metric in SWEEP_METRICS:
            row[metric] = _finite(point.get(metric), metric, unit=metric in SWEEP_METRICS[:3])
        if row["num_groups"] > row["num_samples"] or row["mean_queried_groups"] > row["max_groups"] + 1e-10:
            raise ValueError("inconsistent population or queried-group budget")
        rows.append(row)
        groups[(row["method"], row["cost_weight"], row["max_groups"])].append(row)
    # The legacy producer aggregates these three keys; mixed modes/policy IDs
    # must never be hidden inside an existing aggregate.
    for group in groups.values():
        if len({r["mode"] for r in group}) != 1 or len({r["policy_id"] for r in group}) != 1:
            raise ValueError("mixed modes or policy IDs in summary aggregate")
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("duplicate seed at the same operating point")
    _seed_claim(doc, {r["seed"] for r in rows})
    aggregate = doc.get("aggregate")
    if not isinstance(aggregate, list) or len(aggregate) != len(groups):
        raise ValueError("aggregate must cover every operating point exactly once")
    seen = set()
    for item in aggregate:
        _false_flags(item)
        key = (item.get("method"), _finite(item.get("cost_weight"), "cost_weight"),
               _integer(item.get("max_groups"), "max_groups"))
        if key in seen or key not in groups:
            raise ValueError("duplicate/unmatched aggregate setting")
        seen.add(key)
        if "seeds" not in item or "num_seeds" not in item:
            raise ValueError("aggregate requires explicit seeds and num_seeds")
        _seed_claim(item, {r["seed"] for r in groups[key]})
        for metric, values in _stats(groups[key], SWEEP_METRICS).items():
            for stat, actual in values.items():
                if metric + "_" + stat not in item:
                    raise ValueError(f"aggregate missing {metric}_{stat}")
                supplied = item.get(metric + "_" + stat)
                if actual is None:
                    if supplied is not None:
                        raise ValueError("single-seed SD must be null, not zero")
                elif not math.isclose(_finite(supplied, metric + "_" + stat), actual, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"aggregate {metric}_{stat} differs from source points")
    metadata = {**provenance, "kind": "cost_sweep", "dataset": dataset, "cohort": cohort,
                "split": "validation", "paper_claim": False,
                "split_basis": "explicit_field" if "split" in doc else "versioned_format_contract",
                "dataset_basis": "explicit_field" if "dataset" in doc else "caller_asserted_label",
                "schema_hash": doc.get("schema_hash"), "source_revision": doc.get("source_revision"),
                "backend": doc.get("backend"), "population_identity": "unverified_legacy_summary",
                "source_metrics_rechecked": False,
                "seeds": sorted({r["seed"] for r in rows}), "point_count": len(rows)}
    return rows, metadata


def load_audit(spec):
    (cohort,), directory = _spec(spec, summary=False)
    audit, audit_src = _read(directory / "semantic_audit/audit.json")
    training, train_src = _read(directory / "responder/training.json")
    receipt, receipt_src = _read(directory / "responder/receipt.json")
    schema, schema_src = _read(directory / "responder/schema.json")
    if audit.get("format") != AUDIT_FORMAT or audit.get("status") != "COMPLETED":
        raise ValueError("completed versioned semantic audit required")
    if audit.get("split") != "validation":
        raise ValueError("semantic audits require explicit split=validation")
    _false_flags(audit, require="test_evaluated")
    if audit.get("response_source") != "automatic_model" or audit.get("evidence_status") != "DESCRIPTIVE_SEMANTIC_AUDIT_NOT_PAPER_OR_LATENCY_EVIDENCE":
        raise ValueError("only descriptive automatic-model audits are accepted")
    if training.get("split") != "responder_fit" or training.get("paper_evidence") is not False:
        raise ValueError("training must be responder_fit with paper_evidence=false")
    _false_flags({key: value for key, value in training.items() if key != "split"})
    _false_flags(receipt)
    if training.get("task_label_gradient") is not False:
        raise ValueError("semantic training must record task_label_gradient=false")
    seed = _integer(training.get("seed"), "responder training seed")
    if _integer(receipt.get("seed"), "receipt seed") != seed:
        raise ValueError("training/receipt seed mismatch")
    if receipt.get("data_evidence") != "USER_SUPPLIED_PUBLIC_DATA_NOT_INDEPENDENTLY_AUTHENTICATED":
        raise ValueError("synthetic/unknown data receipts cannot enter research exploratory tables")
    revision = _text(receipt.get("source_revision"), "source_revision")
    if revision.upper().startswith("SYNTHETIC"):
        raise ValueError("synthetic source revision is not research evidence")
    dataset = _text(audit.get("dataset"), "dataset")
    if schema.get("dataset") != dataset:
        raise ValueError("audit/responder schema dataset mismatch")
    schema_hash = _hash(audit.get("schema_hash"), "schema_hash")
    computed_schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False,
                                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if schema.get("schema_version") != "cbmjev-schema-v1" or computed_schema_hash != schema_hash:
        raise ValueError("responder schema contents do not match audit schema_hash")
    if receipt.get("schema_hash") != schema_hash:
        raise ValueError("audit/receipt schema mismatch")
    hashes = audit["source_hashes"]
    for key in ("prepared_samples_sha256", "membership_sha256"):
        if _hash(hashes.get(key), key) != _hash(receipt.get(key), "receipt " + key):
            raise ValueError("audit/responder data population mismatch")
    if _hash(hashes.get("responder_checkpoint_sha256"), "audit checkpoint") != _hash(receipt.get("checkpoint_sha256"), "receipt checkpoint"):
        raise ValueError("audit/responder checkpoint mismatch")
    selection, overall = audit["selection"], audit["overall"]
    if selection.get("audit_population") != "cached_task_labelled_subset_of_requested_split" or selection.get("is_whole_dataset_concept_rate") is not False:
        raise ValueError("unsupported audit population definition")
    sample_ids = _hash(selection.get("audited_sample_ids_hash"), "audited_sample_ids_hash")
    n = _integer(selection.get("audited_cached_split_cases"), "audited cases", 1)
    if overall.get("macro_averaging") != "equal_weight_concepts_with_positive_observed_gold_support" or overall.get("runtime_nonanswers_are_errors_on_observed_gold") is not True:
        raise ValueError("unsupported macro-concept metric semantics")
    supported = sorted(_text(c.get("concept_id"), "concept_id") for c in audit["concepts"]
                       if _integer(c.get("cached_observed_gold_count"), "concept support") > 0)
    if not supported or len(set(supported)) != len(supported) or len(supported) != _integer(overall.get("num_concepts_with_observed_gold"), "supported concepts", 1):
        raise ValueError("invalid/duplicate supported concepts")
    row = {"kind": "semantic_audit", "dataset": dataset, "cohort": cohort,
           "split": "validation", "paper_claim": False, "seed": seed,
           "num_samples": n, "schema_hash": schema_hash, "supported_concepts": supported,
           "audited_sample_ids_hash": sample_ids, "source_revision": revision,
           "prepared_samples_sha256": hashes["prepared_samples_sha256"],
           "membership_sha256": hashes["membership_sha256"],
           "responder_kind": _text(receipt.get("kind"), "responder kind"),
           "initialization": receipt.get("initialization"),
           "training_protocol": {key: training.get(key) for key in (
               "loss", "epochs", "batch_size", "learning_rate", "freeze_backbone",
               "max_length", "pooling", "truncation", "aggregation", "deterministic",
               "n", "observed_labels_per_atom")},
           "semantic_code_hash": receipt.get("semantic_code_hash"),
           "source": audit_src, "training_source": train_src, "receipt_source": receipt_src,
           "schema_source": schema_src, "run": str(directory.resolve())}
    for metric in AUDIT_METRICS:
        row[metric] = _finite(overall.get(metric), metric, unit=True)
    # This is a descriptive responder-seed analysis, not task/controller evidence.
    return row, {"kind": "semantic_audit", "dataset": dataset, "cohort": cohort,
                 "split": "validation", "split_basis": "explicit_field", "paper_claim": False,
                 "paper_claim_basis": "audit_not_paper_evidence_and_training_false",
                 "seed": seed, "artifacts": [audit_src, train_src, receipt_src, schema_src],
                 "responder_checkpoint_link": "audit_to_receipt_verified",
                 "data_authenticity": "user_supplied_not_independently_authenticated"}


def aggregate_rows(rows):
    grouped = defaultdict(list)
    sweep_populations = {}
    for row in rows:
        key = (row["kind"], row["dataset"], row["cohort"])
        if row["kind"] == "cost_sweep":
            population = (row["num_samples"], row["num_groups"], row["cohort_context"])
            if key in sweep_populations and sweep_populations[key] != population:
                raise ValueError("incomparable population or metadata within cost-sweep cohort")
            sweep_populations[key] = population
            key += (row["mode"], row["method"], row["policy_id"], row["cost_weight"], row["max_groups"])
        grouped[key].append(row)
    results = []
    for key, group in sorted(grouped.items()):
        seeds = [r["seed"] for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed in cohort/operating point {key}")
        kind = key[0]
        comparable = ("num_samples", "num_groups") if kind == "cost_sweep" else (
            "num_samples", "schema_hash", "supported_concepts", "audited_sample_ids_hash",
            "prepared_samples_sha256", "membership_sha256", "source_revision",
            "responder_kind", "initialization", "training_protocol", "semantic_code_hash")
        for field in comparable:
            if any(row[field] != group[0][field] for row in group[1:]):
                raise ValueError(f"incomparable {field} within {key}; use distinct cohorts")
        result = {"kind": kind, "dataset": key[1], "cohort": key[2], "split": "validation",
                  "paper_claim": False, "development_only": True,
                  "evidence_label": "SINGLE-SEED DEVELOPMENT-ONLY" if len(seeds) == 1 else "MULTI-SEED DEVELOPMENT-ONLY",
                  "seeds": sorted(seeds), "num_seeds": len(seeds),
                  "seed_unit": "recorded_evaluation_training_seed" if kind == "cost_sweep" else "responder_training_seed",
                  "metrics": _stats(group, SWEEP_METRICS if kind == "cost_sweep" else AUDIT_METRICS),
                  "source_rows": [{"source": r["source"], "row": r.get("source_row"), "seed": r["seed"]} for r in group],
                  **{field: group[0][field] for field in comparable}}
        if kind == "cost_sweep":
            result.update(mode=key[3], method=key[4], policy_id=key[5], cost_weight=key[6], max_groups=key[7])
        results.append(result)
    return results


def latex_escape(value):
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in str(value))


def _cell(metric):
    mean, sd = metric["mean"], metric["sd"]
    return f"{mean:.4f}" if sd is None else f"${mean:.4f} \\pm {sd:.4f}$"


def render_latex(aggregate):
    lines = [r"% Generated from recorded metrics. NOT a formal paper-results table.",
             r"\section*{Exploratory validation results --- development only}",
             r"\noindent\textbf{DEVELOPMENT-ONLY. No formal paper claim is supported or promoted.}",
             r"All recorded operating points are shown; no best-point selection is performed.",
             r"Scores are fractions, not percentages. Values are seed means; $\pm$ denotes sample SD across recorded distinct seeds, not a confidence interval.",
             r"For one seed, SD is undefined and deliberately omitted. Reusing a seed label does not establish independent training.",
             r"Offline replay does not measure latency. Declared costs have no inferred wall-clock units."]
    lines += [r"Changing $\lambda$ changes the decision penalty, not the measured backend cost regime; these tables do not establish an error-structure mechanism."]
    blocks = defaultdict(list)
    for item in aggregate:
        blocks[(item["kind"], item["dataset"], item["cohort"], item.get("mode", "audit"))].append(item)
    for (kind, dataset, cohort, mode), rows in sorted(blocks.items()):
        seeds = sorted({seed for row in rows for seed in row["seeds"]})
        warning = "SINGLE-SEED DEVELOPMENT-ONLY" if len(seeds) == 1 else "MULTI-SEED DEVELOPMENT-ONLY"
        lines.extend([r"\subsection*{" + latex_escape(f"{dataset} / {cohort} / {mode}") + "}",
                      r"\noindent\textbf{" + warning + "}. Recorded seeds: " + latex_escape(", ".join(map(str, seeds))) + "."])
        if kind == "cost_sweep":
            lines += [r"Population identity and dataset label may be unverified in legacy summaries; see the JSON manifest.",
                      r"Distinct seeds are counted separately for each operating point ($n$); sweep rows are not replications.",
                      r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{llrrrllll}",
                      r"Method & Policy & Budget & $\lambda$ & $n$ & Accuracy & Macro-F1 & Groups & Cost \\",
                      r"\hline\endhead"]
            for row in rows:
                fields = [latex_escape(row["method"]), latex_escape(row["policy_id"]), str(row["max_groups"]),
                          f"{row['cost_weight']:g}", str(row["num_seeds"])]
                fields += [_cell(row["metrics"][metric]) for metric in ("accuracy", "macro_f1", "mean_queried_groups", "mean_declared_cost")]
                lines.append(" & ".join(fields) + r" \\")
            lines += [r"\end{longtable}}", r"Group-mean risk and call counts are retained in the JSON; calls are not measured runtime."]
        else:
            lines += [r"Cached task-labelled validation subset only; macro means equally weight concepts with positive observed-gold support.",
                      r"Runtime nonanswers count as errors. These are responder metrics, not task accuracy or controller evidence.",
                      r"\begin{longtable}{lrrll}",
                      r"Responder & Cases & Seeds & Macro concept accuracy & Macro concept F1 \\", r"\hline\endhead"]
            for row in rows:
                fields = [latex_escape(row["responder_kind"]), str(row["num_samples"]), str(row["num_seeds"])]
                fields += [_cell(row["metrics"][metric]) for metric in AUDIT_METRICS]
                lines.append(" & ".join(fields) + r" \\")
            lines += [r"\end{longtable}", r"Matching artifact hashes establish association, not independent data-authenticity or pretraining-overlap verification."]
    return "\n".join(lines) + "\n"


def build(summary_specs, audit_specs, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output must be a new directory: {out}")
    rows, sources = [], []
    for spec in summary_specs:
        loaded, metadata = load_summary(spec)
        rows.extend(loaded)
        sources.append(metadata)
    for spec in audit_specs:
        row, metadata = load_audit(spec)
        rows.append(row)
        sources.append(metadata)
    if not rows:
        raise ValueError("at least one summary or audit run is required")
    aggregate = aggregate_rows(rows)
    limitations = [
        "Validation development evidence only; no formal table or Pending result is replaced.",
        "Distinct seed labels are not proof of independent training; shared-responder conditioning may remain.",
        "Legacy summary dataset labels are caller assertions; population identity/raw metrics may be unavailable.",
        "No best-setting selection, significance test, confidence interval, strong-AFA equivalence, or causal claim.",
        "Offline replay does not establish latency, throughput, or avoided perception cost.",
        "No independent experiment-integrity audit is performed by this converter.",
    ]
    claims = {"C1": {"claim_supported": "no", "scope": "measurement-error structure x real/shared cost regimes",
                     "reason": "Descriptive sweep or responder metrics alone do not establish the required controlled mechanism evidence."},
              "C2": {"claim_supported": "no", "scope": "attributable decision benefit against matched strong active feature acquisition",
                     "reason": "This export does not verify matched strong baselines, held-out confirmation, or multi-dataset benefit."}}
    evidence = {"format": "cbmjev-development-claim-evidence-v1", "split": "validation",
                "paper_claim": False, "development_only": True, "formal_results_modified": False,
                "claim_gate": "NOT_PROMOTED_BY_DESIGN", "claims": claims,
                "integrity_status": "unavailable_not_audited_by_converter",
                "provisional": True, "limitations": limitations,
                "aggregation": "unweighted mean and sample SD across distinct recorded seeds per dataset/cohort/operating point",
                "rows": rows, "aggregate": aggregate}
    manifest = {"format": "cbmjev-development-export-v1", "paper_claim": False,
                "development_only": True, "sources": sources,
                "point_count": len(rows), "aggregate_count": len(aggregate),
                "seed_coverage": [{key: item[key] for key in ("kind", "dataset", "cohort", "seeds", "num_seeds")}
                                  | {key: item[key] for key in ("method", "policy_id", "cost_weight", "max_groups", "mode") if key in item}
                                  for item in aggregate],
                "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "invocation": {"summaries": list(summary_specs), "audit_runs": list(audit_specs)},
                "rounding": "LaTeX: four decimals; JSON: source precision/derived full precision",
                "uncertainty": "sample SD (ddof=1), null for one seed; no CI or significance claim"}
    document = ("\\documentclass[10pt]{article}\n\\usepackage[T1]{fontenc}\n"
                "\\usepackage[margin=0.65in]{geometry}\n\\usepackage{longtable}\n"
                "\\begin{document}\n\\input{exploratory_tables.tex}\n\\end{document}\n")
    files = {"claim_evidence.json": json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
             "exploratory_tables.tex": render_latex(aggregate), "exploratory_report.tex": document}
    manifest["outputs"] = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()}
    files["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # All validation precedes any output creation; never overwrite an earlier export.
    out.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (out / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], metavar="DATASET/COHORT=PATH",
                        help="repeat for distinct seeds/datasets; same cohort explicitly asserts comparable protocol")
    parser.add_argument("--audit-run", action="append", default=[], metavar="COHORT=RUN_DIR",
                        help="requires semantic_audit/audit.json and responder/{training,receipt,schema}.json")
    parser.add_argument("--out", required=True, help="new output directory; no overwrite")
    args = parser.parse_args()
    try:
        manifest = build(args.summary, args.audit_run, args.out)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"out": str(Path(args.out).resolve()), "points": manifest["point_count"],
                      "aggregate_rows": manifest["aggregate_count"], "paper_claim": False}))


if __name__ == "__main__":
    main()
