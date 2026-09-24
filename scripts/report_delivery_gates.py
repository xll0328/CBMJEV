#!/usr/bin/env python3
"""Report lightweight CBMJev paper/project readiness gates.

This is a fast dashboard, not a release audit.  It only checks the presence of
paper-facing evidence artifacts and the explicit flags in
paper/cvpr2027/evidence_state.json so that sprint decisions can be made without
rerunning expensive experiments or validating large artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Gate:
    gate_id: str
    label: str
    passed: bool
    evidence: list[str]
    missing: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "label": self.label,
            "passed": self.passed,
            "evidence": self.evidence,
            "missing": self.missing,
            "note": self.note,
        }


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _gate_files(
    root: Path,
    gate_id: str,
    label: str,
    required: list[str],
    note_pass: str,
    note_fail: str,
) -> Gate:
    missing = [rel for rel in required if not _exists(root, rel)]
    evidence = [rel for rel in required if rel not in missing]
    return Gate(
        gate_id=gate_id,
        label=label,
        passed=not missing,
        evidence=evidence,
        missing=missing,
        note=note_pass if not missing else note_fail,
    )


def build_report(root: Path) -> dict[str, Any]:
    evidence_state_path = root / "paper/cvpr2027/evidence_state.json"
    evidence_state = _load_json(evidence_state_path)

    gates: list[Gate] = []

    gates.append(
        _gate_files(
            root,
            "paper_compile_artifacts",
            "Paper PDFs exist",
            [
                "paper/cvpr2027/build/arxiv/arxiv.pdf",
                "paper/cvpr2027/build/review/review.pdf",
                "paper/cvpr2027/BUILD_REPORT.md",
            ],
            "Both paper variants have compiled artifacts recorded.",
            "Compile arXiv/review variants before claiming a paper handoff.",
        )
    )
    gates.append(
        _gate_files(
            root,
            "development_evidence_imported",
            "Development evidence imported",
            [
                "results/main/cub_seed60_budget_grid/metrics.json",
                "results/main/boundary_evidence/summary.json",
                "results/main/cub_live_profile_61_62/summary.json",
                "paper/cvpr2027/generated/boundary_evidence_table.tex",
                "paper/cvpr2027/generated/cub_live_profile_table.tex",
            ],
            "Current single-seed/boundary/cost-accounting evidence is present.",
            "Some already-claimed development evidence artifacts are missing.",
        )
    )
    gates.append(
        _gate_files(
            root,
            "cub_value_policy_60_61_62",
            "CUB value-policy aggregate staged",
            [
                "results/main/cub_value_budget_grid_60_61_62/frontier_long.csv",
                "results/main/cub_value_budget_grid_60_61_62/frontier_summary.json",
                "results/main/cub_value_budget_grid_60_61_62/claim_analysis_accuracy.md",
                "results/main/cub_value_budget_grid_60_61_62/claim_analysis_macro_f1.md",
                "results/main/cub_value_budget_grid_60_61_62/paired_seed_deltas_accuracy.md",
                "results/main/cub_value_budget_grid_60_61_62/paired_seed_deltas_macro_f1.md",
                "paper/cvpr2027/generated/cub_value_policy_manifest.json",
                "paper/cvpr2027/generated/cub_value_policy_table.tex",
                "results/main/cub_value_policy_paper_update_60_61_62.json",
                "results/main/CUB_VALUE_POLICY_PAPER_UPDATE_60_61_62.md",
            ],
            "The first three-seed CUB value-policy aggregate is staged with wording guardrails.",
            "The core dynamic value-policy claim remains pending.",
        )
    )
    gates.append(
        _gate_files(
            root,
            "theory_alignment",
            "Theory-to-implementation alignment exists",
            [
                "theory/THEORY_APPENDIX.md",
                "theory/PROOF_OBLIGATION_LEDGER.md",
                "theory/THEORY_IMPLEMENTATION_ALIGNMENT_20260923.md",
            ],
            "Theory files and implementation-alignment audit are present.",
            "Theory package is incomplete or not aligned to implementation.",
        )
    )
    gates.append(
        _gate_files(
            root,
            "reproducibility_entrypoints",
            "Reproducibility entrypoints documented",
            [
                "README.md",
                "START_HERE.md",
                "docs/RUN_AUTOMATION_20260923.md",
                "docs/FINAL_DELIVERY_INDEX_DRAFT_20260923.md",
                "paper/CLAIM_EVIDENCE_MATRIX.md",
            ],
            "Primary human handoff documents are present.",
            "The project is missing one or more handoff documents.",
        )
    )

    boolean_flags = {
        "empirical_results_promoted": bool(evidence_state.get("empirical_results_promoted")),
        "three_seed_matched_baselines_audited": bool(
            evidence_state.get("three_seed_matched_baselines_audited")
        ),
        "live_cost_claim_supported": bool(evidence_state.get("live_cost_claim_supported")),
        "cvpr2027_format_reverified": bool(evidence_state.get("cvpr2027_format_reverified")),
        "complete_citation_audit": bool(evidence_state.get("complete_citation_audit")),
    }

    submission_ready = (
        all(g.passed for g in gates)
        and all(boolean_flags.values())
        and evidence_state.get("status") == "SUBMISSION_READY"
    )

    return {
        "format": "cbmjev-delivery-gates-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "evidence_state_path": str(evidence_state_path),
        "evidence_state_status": evidence_state.get("status", "MISSING"),
        "boolean_flags": boolean_flags,
        "gates": [g.to_dict() for g in gates],
        "summary": {
            "passed_gates": sum(1 for g in gates if g.passed),
            "total_gates": len(gates),
            "submission_ready": submission_ready,
            "paper_arxiv_candidate": gates[0].passed and gates[1].passed,
            "core_dynamic_claim_ready": gates[2].passed
            and bool(evidence_state.get("empirical_results_promoted")),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CBMJev delivery gates",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        "",
        f"Evidence state: `{report['evidence_state_status']}`",
        "",
        "| Gate | Status | Missing | Note |",
        "|---|---:|---|---|",
    ]
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "PENDING"
        missing = ", ".join(f"`{m}`" for m in gate["missing"]) if gate["missing"] else "—"
        lines.append(f"| {gate['label']} | {status} | {missing} | {gate['note']} |")
    lines.extend(
        [
            "",
            "## Summary flags",
            "",
            "```json",
            json.dumps(report["summary"], indent=2, sort_keys=True),
            "```",
            "",
            "This file is a fast sprint dashboard, not a final release audit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    report = build_report(root)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    if not args.out_json and not args.out_md:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
