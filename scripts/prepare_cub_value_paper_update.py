#!/usr/bin/env python3
"""Prepare paper/evidence updates after a CUB value-policy aggregate is staged.

The script is intentionally conservative.  It reads the completed aggregate and
claim-analysis files, then writes a machine-readable recommendation and a human
Markdown checklist.  It does not rewrite the paper text or promote a claim by
itself.  With ``--apply-evidence-state`` it only appends/updates one development
evidence entry in ``paper/cvpr2027/evidence_state.json``; global readiness flags
remain false unless a human/release process changes them.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_WORDING = {
    "candidate_positive_validation_claim": {
        "claim_level": "validation_scoped_positive",
        "allowed": (
            "On CUB validation, the dynamic value policy improves over the best "
            "explicitly listed matched baseline at multiple matched budgets."
        ),
        "forbidden": (
            "Do not call this a test-set result, deployment speedup, medical "
            "claim, official Jev result, or final CVPR-ready empirical claim."
        ),
    },
    "weak_or_budget_local_dynamic_gain": {
        "claim_level": "narrow_operating_region",
        "allowed": (
            "On CUB validation, dynamic value shows a local/budget-specific "
            "advantage that motivates focused follow-up."
        ),
        "forbidden": "Do not write broad superiority over static masks or AFA baselines.",
    },
    "negative_or_static_equivalent": {
        "claim_level": "boundary_or_negative",
        "allowed": (
            "On CUB validation, matched static/fixed controls are competitive "
            "with or better than dynamic value policies under the tested setup."
        ),
        "forbidden": "Do not hide the negative result or present query adaptivity as proven beneficial.",
    },
    "no_observed_matched_budget_gain": {
        "claim_level": "descriptive_boundary_not_equivalence",
        "allowed": (
            "On CUB validation, the selected dynamic value policies show no "
            "matched-budget gain over the explicitly listed baselines in "
            "this development grid."
        ),
        "forbidden": (
            "Do not claim statistical equivalence, inferiority, an independent "
            "canonical fixed comparison when fixed aliases static, or a general JEV failure."
        ),
    },
    "tie_or_uninformative": {
        "claim_level": "inconclusive",
        "allowed": "Report the aggregate as inconclusive validation evidence.",
        "forbidden": "Do not promote a dynamic-value advantage.",
    },
    "insufficient_dynamic_or_static_rows": {
        "claim_level": "incomplete_evidence",
        "allowed": "Report that the aggregate is structurally incomplete for the dynamic-vs-static claim.",
        "forbidden": "Do not infer any dynamic-policy claim.",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def best_delta(report: dict[str, Any]) -> float | None:
    deltas = [
        row.get("delta_vs_best_baseline", row.get("delta_vs_best_static_or_fixed"))
        for row in report.get("comparisons", [])
        if isinstance(row.get("delta_vs_best_baseline", row.get("delta_vs_best_static_or_fixed")), (int, float))
    ]
    return max(deltas) if deltas else None


def build_recommendation(root: Path, aggregate_dir: Path, paper_dir: Path, tag: str) -> dict[str, Any]:
    summary_path = require_file(aggregate_dir / "frontier_summary.json")
    long_path = require_file(aggregate_dir / "frontier_long.csv")
    acc_path = require_file(aggregate_dir / "claim_analysis_accuracy.json")
    f1_path = require_file(aggregate_dir / "claim_analysis_macro_f1.json")
    paired_acc_path = require_file(aggregate_dir / "paired_seed_deltas_accuracy.json")
    paired_f1_path = require_file(aggregate_dir / "paired_seed_deltas_macro_f1.json")
    manifest_path = require_file(paper_dir / "generated" / f"{tag}_manifest.json")
    table_path = require_file(paper_dir / "generated" / f"{tag}_table.tex")

    summary = load_json(summary_path)
    acc = load_json(acc_path)
    f1 = load_json(f1_path)
    paired_acc = load_json(paired_acc_path)
    paired_f1 = load_json(paired_f1_path)
    manifest = load_json(manifest_path)

    acc_status = acc.get("status", "missing")
    f1_status = f1.get("status", "missing")
    acc_wording = STATUS_WORDING.get(acc_status, STATUS_WORDING["tie_or_uninformative"])
    f1_wording = STATUS_WORDING.get(f1_status, STATUS_WORDING["tie_or_uninformative"])

    evidence_id = f"cub_value_policy_{'_'.join(str(s) for s in summary.get('seeds', []))}_v1"
    return {
        "format": "cbmjev-cub-value-paper-update-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "aggregate_dir": str(aggregate_dir),
        "paper_dir": str(paper_dir),
        "tag": tag,
        "evidence_entry": {
            "id": evidence_id,
            "scope": "CUB validation multi-seed value-policy offline replay",
            "metrics": str(long_path.relative_to(root) if long_path.is_relative_to(root) else long_path),
            "summary": str(summary_path.relative_to(root) if summary_path.is_relative_to(root) else summary_path),
            "claim_analysis_accuracy": str(acc_path.relative_to(root) if acc_path.is_relative_to(root) else acc_path),
            "claim_analysis_macro_f1": str(f1_path.relative_to(root) if f1_path.is_relative_to(root) else f1_path),
            "paired_seed_deltas_accuracy": str(paired_acc_path.relative_to(root) if paired_acc_path.is_relative_to(root) else paired_acc_path),
            "paired_seed_deltas_macro_f1": str(paired_f1_path.relative_to(root) if paired_f1_path.is_relative_to(root) else paired_f1_path),
            "table": str(table_path.relative_to(paper_dir) if table_path.is_relative_to(paper_dir) else table_path),
            "manifest": str(manifest_path.relative_to(paper_dir) if manifest_path.is_relative_to(paper_dir) else manifest_path),
            "paper_claim_status": "validation_scoped_pending_manual_wording",
        },
        "summary": {
            "seeds": summary.get("seeds"),
            "num_seeds": summary.get("num_seeds"),
            "num_rows": summary.get("num_rows"),
            "accuracy_status": acc_status,
            "macro_f1_status": f1_status,
            "paired_accuracy_status": paired_acc.get("status"),
            "paired_macro_f1_status": paired_f1.get("status"),
            "accuracy_best_delta": best_delta(acc),
            "macro_f1_best_delta": best_delta(f1),
            "accuracy_claim_level": acc_wording["claim_level"],
            "macro_f1_claim_level": f1_wording["claim_level"],
            "accuracy_baseline_methods": acc.get("baseline_methods"),
            "macro_f1_baseline_methods": f1.get("baseline_methods"),
        },
        "wording_guardrails": {
            "accuracy": acc_wording,
            "macro_f1": f1_wording,
            "always_forbidden": [
                "test-set claim",
                "deployment/runtime speedup from offline replay",
                "official Jev/NanoJev claim",
                "zero-shot or label-free claim",
                "medical reliability claim",
                "CVPR-final readiness without matched strong baselines and audit",
            ],
        },
        "paper_files_to_review": [
            "paper/cvpr2027/sections/experiments.tex",
            "paper/cvpr2027/evidence_state.json",
            "paper/CLAIM_EVIDENCE_MATRIX.md",
            "results/main/SPRINT_STATUS_20260923.md",
            "paper/cvpr2027/BUILD_REPORT.md",
        ],
        "source_manifest": manifest,
    }


def update_evidence_state(path: Path, evidence_entry: dict[str, Any]) -> dict[str, Any]:
    state = load_json(path)
    updated = deepcopy(state)
    rows = list(updated.get("development_results", []))
    rows = [row for row in rows if row.get("id") != evidence_entry["id"]]
    rows.append(evidence_entry)
    updated["development_results"] = rows
    updated["development_results_imported"] = True
    updated["empirical_results_promoted"] = False
    updated["three_seed_matched_baselines_audited"] = False
    updated["status"] = "DRAFT_NOT_SUBMISSION_READY"
    updated["note"] = (
        "CUB value-policy aggregate may be imported as validation development "
        "evidence, but final claims still require manual wording, matched strong "
        "baselines, locked test evaluation, isolated speed profiling, and audit."
    )
    path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return updated


def write_markdown(report: dict[str, Any], path: Path) -> None:
    s = report["summary"]
    lines = [
        "# CUB value-policy paper update recommendation",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        "",
        "## Claim-analysis summary",
        "",
        f"- Seeds: `{s.get('seeds')}`",
        f"- Accuracy status: `{s.get('accuracy_status')}`; baselines: `{s.get('accuracy_baseline_methods')}`; best dynamic-minus-baseline delta: `{s.get('accuracy_best_delta')}`",
        f"- Macro-F1 status: `{s.get('macro_f1_status')}`; baselines: `{s.get('macro_f1_baseline_methods')}`; best dynamic-minus-baseline delta: `{s.get('macro_f1_best_delta')}`",
        f"- Paired seed accuracy status: `{s.get('paired_accuracy_status')}`",
        f"- Paired seed Macro-F1 status: `{s.get('paired_macro_f1_status')}`",
        "",
        "## Allowed wording direction",
        "",
        f"- Accuracy: {report['wording_guardrails']['accuracy']['allowed']}",
        f"- Macro-F1: {report['wording_guardrails']['macro_f1']['allowed']}",
        "",
        "## Always forbidden",
        "",
    ]
    lines.extend(f"- {item}" for item in report["wording_guardrails"]["always_forbidden"])
    lines.extend([
        "",
        "## Evidence entry to import",
        "",
        "```json",
        json.dumps(report["evidence_entry"], indent=2, sort_keys=True),
        "```",
        "",
        "## Files to review after import",
        "",
    ])
    lines.extend(f"- `{item}`" for item in report["paper_files_to_review"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper/cvpr2027"))
    parser.add_argument("--tag", default="cub_value_policy")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--apply-evidence-state", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    aggregate_dir = (root / args.aggregate_dir).resolve() if not args.aggregate_dir.is_absolute() else args.aggregate_dir
    paper_dir = (root / args.paper_dir).resolve() if not args.paper_dir.is_absolute() else args.paper_dir
    report = build_recommendation(root, aggregate_dir, paper_dir, args.tag)

    if args.apply_evidence_state:
        state_path = paper_dir / "evidence_state.json"
        updated_state = update_evidence_state(state_path, report["evidence_entry"])
        report["applied_evidence_state"] = str(state_path)
        report["evidence_state_development_results"] = len(updated_state.get("development_results", []))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, args.out_md)
    print(json.dumps({
        "wrote": [str(args.out_json), str(args.out_md)],
        "accuracy_status": report["summary"]["accuracy_status"],
        "macro_f1_status": report["summary"]["macro_f1_status"],
        "applied_evidence_state": report.get("applied_evidence_state"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
