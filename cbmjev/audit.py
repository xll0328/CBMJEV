"""Side-channel-only semantic audits of public concept labels vs cached responses.

No gold labels or annotation missingness are returned to a responder, classifier,
or acquisition policy. This module never fits models or chooses thresholds.
"""

import math
import statistics
from collections import Counter
from pathlib import Path

from .contracts import stable_hash
from .io import file_hash, fresh_dir, write_json


def _mean_or_none(values):
    return statistics.mean(values) if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _gold_values(record, schema):
    entries = record.get("concepts")
    if not isinstance(entries, list):
        raise ValueError("Canonical concepts must be a list.")
    indexed = {}
    expected = {concept.id for concept in schema.concepts}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("concept_id") in indexed:
            raise ValueError("Malformed or duplicated canonical concept labels.")
        if entry.get("concept_id") not in expected:
            raise ValueError("Canonical concept is absent from schema.")
        indexed[entry["concept_id"]] = entry
    if set(indexed) != expected:
        raise ValueError("Canonical concept IDs must exactly match the schema.")
    values, statuses = [], []
    for concept in schema.concepts:
        entry = indexed[concept.id]
        status, value = entry.get("annotation_status"), entry.get("value")
        if not isinstance(status, str) or not status:
            raise ValueError("Canonical annotation_status must be a nonempty string.")
        if status == "OBSERVED":
            if type(value) is not int or not 0 <= value < len(concept.values):
                raise ValueError("OBSERVED gold must be a semantic category, not a runtime status.")
            values.append(value)
        else:
            if value is not None:
                raise ValueError("Non-OBSERVED annotation must not carry a scored gold value.")
            values.append(None)
        statuses.append(status)
    return tuple(values), tuple(statuses)


def _coupling(first, second):
    common = sorted(set(first).intersection(second))
    counts = Counter((first[sample], second[sample]) for sample in common)
    n00, n01 = counts[(0, 0)], counts[(0, 1)]
    n10, n11 = counts[(1, 0)], counts[(1, 1)]
    variance_product = (n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10)
    phi = (n11 * n00 - n10 * n01) / math.sqrt(variance_product) if variance_product else None
    return {
        "paired_n": len(common),
        "n_both_correct": n00,
        "n_first_correct_second_error": n01,
        "n_first_error_second_correct": n10,
        "n_both_error": n11,
        "phi": phi,
        "phi_undefined_reason": (
            None if variance_product else
            "NO_PAIRED_GOLD" if not common else "ZERO_ERROR_INDICATOR_VARIANCE"
        ),
    }


def audit_responses(prepared, cache, out, *, split="validation", evaluate_test=False):
    """Write audit.json in a new/empty output directory and return its report.

    Primary accuracy/F1 count runtime UNCERTAIN/NOT_APPLICABLE as incorrect
    nonanswers; a semantic category named 'unknown' remains a normal answer.
    Group error means any atom wrong, on cases with all group atoms gold-observed.
    """
    if type(evaluate_test) is not bool:
        raise ValueError("evaluate_test must be a boolean.")
    if split not in {"validation", "test"}:
        raise ValueError("Semantic quality audit only accepts validation or explicitly enabled test.")
    if split == "test" and not evaluate_test:
        raise ValueError("Test audit requires evaluate_test=True; never tune on test labels.")
    # Import here: pipeline owns artifact loading; no model-fitting dependency
    # is exercised by this function and future CLI imports do not form a cycle.
    from .pipeline import load_cache, load_prepared

    prepared, cache = Path(prepared), Path(cache)
    schema, canonical, membership = load_prepared(prepared)
    cache_schema, cached, manifest = load_cache(cache)
    if schema.hash != cache_schema.hash:
        raise ValueError("Prepared and cache schemas differ.")
    prepared_hash = file_hash(prepared / "samples.jsonl")
    membership_hash = file_hash(prepared / "membership.jsonl")
    if manifest.get("prepared_samples_sha256") != prepared_hash:
        raise ValueError("Cache is not linked to these canonical prepared samples.")
    if manifest.get("membership_sha256") != membership_hash:
        raise ValueError("Cache was generated under a different split membership.")
    source = manifest.get("response_source")
    if source not in {"automatic_model", "synthetic"}:
        raise ValueError("Audit expects automatic model responses or explicitly synthetic fixtures.")
    canonical_by_id = {row["sample_id"]: row for row in canonical}
    cached_by_id = {}
    for row in cached:
        sample = row["sample_id"]
        if sample not in canonical_by_id:
            raise ValueError("Cache contains a sample absent from prepared data.")
        original = canonical_by_id[sample]
        if row["group_id"] != original["group_id"] or row["split"] != membership[sample]["split"]:
            raise ValueError("Cache group/split differs from canonical membership.")
        target = original.get("target", {})
        if target.get("status") != "OBSERVED" or type(target.get("value")) is not int:
            raise ValueError("Cache includes a case without an observed canonical task target.")
        if row["y"] != target["value"]:
            raise ValueError("Cache and canonical task targets differ.")
        if row.get("response_source", source) != source:
            raise ValueError("Cache row response_source differs from its manifest.")
        cached_by_id[sample] = row
    canonical_selected = [
        row for row in canonical if membership[row["sample_id"]]["split"] == split
    ]
    selected = [
        cached_by_id[row["sample_id"]]
        for row in canonical_selected if row["sample_id"] in cached_by_id
    ]
    gold, annotation_statuses = {}, {}
    for row in canonical_selected:
        gold[row["sample_id"]], annotation_statuses[row["sample_id"]] = _gold_values(row, schema)
    n_cached = len(selected)
    concepts = []
    for atom, concept in enumerate(schema.concepts):
        categories = len(concept.values)
        # Runtime nonanswers have their own columns, never new semantic classes.
        confusion = [[0] * (categories + 2) for _ in range(categories)]
        skipped = Counter()
        runtime_all, runtime_gold = Counter(), Counter()
        for row in selected:
            sample, response = row["sample_id"], row["z"][atom]
            if response == categories:
                runtime_all["UNCERTAIN"] += 1
            elif response == categories + 1:
                runtime_all["NOT_APPLICABLE"] += 1
            label = gold[sample][atom]
            if label is None:
                skipped[annotation_statuses[sample][atom]] += 1
                continue
            confusion[label][response] += 1
            if response == categories:
                runtime_gold["UNCERTAIN"] += 1
            elif response == categories + 1:
                runtime_gold["NOT_APPLICABLE"] += 1
        supports = [sum(row) for row in confusion]
        support = sum(supports)
        correct = sum(confusion[value][value] for value in range(categories))
        answered = sum(sum(row[:categories]) for row in confusion)
        class_f1 = []
        for value in range(categories):
            tp = confusion[value][value]
            fp = sum(confusion[other][value] for other in range(categories) if other != value)
            # Includes runtime nonanswers as false negatives.
            fn = supports[value] - tp
            denominator = 2 * tp + fp + fn
            class_f1.append(2 * tp / denominator if denominator else 0.0)
        canonical_support = sum(
            gold[row["sample_id"]][atom] is not None for row in canonical_selected
        )
        recalls = [_ratio(confusion[value][value], supports[value])
                   for value in range(categories)]
        concepts.append({
            "concept_id": concept.id,
            "semantic_values": list(concept.values),
            "num_semantic_classes_fixed": categories,
            "canonical_split_observed_gold_count": canonical_support,
            "cached_observed_gold_count": support,
            "cached_missing_gold_count": n_cached - support,
            "cached_gold_annotation_statuses_not_scored": dict(sorted(skipped.items())),
            "correct_semantic_count": correct,
            "semantic_answer_count_on_observed_gold": answered,
            "runtime_nonanswer_count_on_observed_gold": support - answered,
            "runtime_uncertain_count_on_observed_gold": runtime_gold["UNCERTAIN"],
            "runtime_not_applicable_count_on_observed_gold": runtime_gold["NOT_APPLICABLE"],
            "runtime_uncertain_count_all_cached": runtime_all["UNCERTAIN"],
            "runtime_not_applicable_count_all_cached": runtime_all["NOT_APPLICABLE"],
            "accuracy_runtime_nonanswers_count_as_errors": _ratio(correct, support),
            "macro_f1_fixed_semantic_classes": statistics.mean(class_f1) if support else None,
            "coverage_semantic_answers_on_observed_gold": _ratio(answered, support),
            "conditional_accuracy_among_semantic_answers": _ratio(correct, answered),
            "per_class_gold_support": supports,
            "per_class_recall_missing_support_null": recalls,
            "balanced_accuracy_supported_semantic_classes": _mean_or_none(
                [recall for recall in recalls if recall is not None]),
            "num_supported_semantic_classes": sum(count > 0 for count in supports),
            "descriptive_majority_fraction_on_audited_gold": _ratio(max(supports), support),
            "majority_fraction_scope": "heldout label distribution only; not a training-selected baseline",
            "per_class_f1_zero_division_zero": class_f1 if support else [None] * categories,
            "confusion_gold_rows_response_columns": confusion,
            "response_column_labels": (
                ["semantic:" + value for value in concept.values]
                + ["runtime:UNCERTAIN", "runtime:NOT_APPLICABLE"]
            ),
        })
    group_errors, group_reports = {}, []
    for group in schema.groups:
        errors, answered = {}, 0
        for row in selected:
            sample = row["sample_id"]
            if not all(gold[sample][atom] is not None for atom in group.atoms):
                continue
            errors[sample] = int(any(row["z"][atom] != gold[sample][atom] for atom in group.atoms))
            answered += int(all(row["z"][atom] < len(schema.concepts[atom].values) for atom in group.atoms))
        canonical_complete = sum(
            all(gold[row["sample_id"]][atom] is not None for atom in group.atoms)
            for row in canonical_selected
        )
        group_errors[group.id] = errors
        group_reports.append({
            "query_group_id": group.id,
            "concept_ids": [schema.concepts[atom].id for atom in group.atoms],
            "canonical_split_complete_gold_cases": canonical_complete,
            "cached_complete_gold_cases": len(errors),
            "cached_incomplete_gold_cases_not_scored": n_cached - len(errors),
            "group_error_count": sum(errors.values()),
            "group_error_rate": _ratio(sum(errors.values()), len(errors)),
            "group_semantic_answer_coverage": _ratio(answered, len(errors)),
        })
    supported = [item for item in concepts if item["cached_observed_gold_count"]]
    total_support = sum(item["cached_observed_gold_count"] for item in concepts)
    total_correct = sum(item["correct_semantic_count"] for item in concepts)
    total_answered = sum(item["semantic_answer_count_on_observed_gold"] for item in concepts)
    selected_ids = sorted(row["sample_id"] for row in selected)
    uncached_ids = sorted(
        row["sample_id"] for row in canonical_selected if row["sample_id"] not in cached_by_id
    )
    report = {
        "format": "cbmjev-semantic-response-audit-v1",
        "status": (
            "NO_CACHED_CASES" if not selected else
            "NO_OBSERVED_CONCEPT_GOLD" if not supported else "COMPLETED"
        ),
        "dataset": schema.dataset,
        "split": split,
        "test_evaluated": split == "test",
        "response_source": source,
        "evidence_status": (
            "SYNTHETIC_FIXTURE_NOT_RESEARCH_EVIDENCE" if source == "synthetic"
            else "DESCRIPTIVE_SEMANTIC_AUDIT_NOT_PAPER_OR_LATENCY_EVIDENCE"
        ),
        "schema_hash": schema.hash,
        "source_hashes": {
            "prepared_samples_sha256": prepared_hash,
            "membership_sha256": membership_hash,
            "prepared_schema_file_sha256": file_hash(prepared / "schema.json"),
            "cache_responses_sha256": file_hash(cache / "responses.jsonl"),
            "cache_manifest_sha256": file_hash(cache / "manifest.json"),
            "responder_checkpoint_sha256": manifest.get("responder_checkpoint_sha256"),
        },
        "selection": {
            "audit_population": "cached_task_labelled_subset_of_requested_split",
            "is_whole_dataset_concept_rate": False,
            "canonical_total_cases_all_roles": len(canonical),
            "cache_total_cases_all_roles": len(cached),
            "canonical_total_not_cached": len(canonical) - len(cached),
            "total_uncached_may_include_responder_fit_and_missing_task_labels": True,
            "canonical_split_cases": len(canonical_selected),
            "canonical_split_observed_task_target_cases": sum(
                row["target"]["status"] == "OBSERVED" for row in canonical_selected
            ),
            "canonical_split_missing_task_target_cases": sum(
                row["target"]["status"] != "OBSERVED" for row in canonical_selected
            ),
            "audited_cached_split_cases": n_cached,
            "canonical_split_not_cached": len(uncached_ids),
            "canonical_split_observed_target_not_cached": sum(
                row["sample_id"] not in cached_by_id and row["target"]["status"] == "OBSERVED"
                for row in canonical_selected
            ),
            "audited_sample_ids_hash": stable_hash(selected_ids),
            "uncached_sample_ids_hash": stable_hash(uncached_ids),
        },
        "overall": {
            "num_concepts_total": schema.num_atoms,
            "num_concepts_with_observed_gold": len(supported),
            "num_concepts_without_observed_gold": schema.num_atoms - len(supported),
            "num_scored_case_concept_pairs": total_support,
            "macro_concept_accuracy": _mean_or_none([
                item["accuracy_runtime_nonanswers_count_as_errors"] for item in supported
            ]),
            "macro_concept_f1": _mean_or_none([
                item["macro_f1_fixed_semantic_classes"] for item in supported
            ]),
            "macro_concept_coverage": _mean_or_none([
                item["coverage_semantic_answers_on_observed_gold"] for item in supported
            ]),
            "micro_concept_accuracy": _ratio(total_correct, total_support),
            "micro_concept_coverage": _ratio(total_answered, total_support),
            "macro_averaging": "equal_weight_concepts_with_positive_observed_gold_support",
            "unsupported_concepts_are_null_not_zero": True,
            "runtime_nonanswers_are_errors_on_observed_gold": True,
            "semantic_unknown_is_an_ordinary_schema_category": True,
        },
        "concepts": concepts,
        "query_groups": group_reports,
        "group_error_coupling": {
            "query_group_ids": [group.id for group in schema.groups],
            "error_definition": "any_atom_wrong_in_query_group_runtime_nonanswers_are_wrong",
            "gold_support_rule": "all_atoms_in_both_query_groups_have_OBSERVED_gold",
            "counting_unit": "cached_case_not_independent_patient_or_edit_family",
            "is_causal_relationship": False,
            "independence_or_uncertainty_interval_claimed": False,
            "matrix": [
                [_coupling(group_errors[first.id], group_errors[second.id]) for second in schema.groups]
                for first in schema.groups
            ],
        },
        "information_boundary": {
            "gold_and_missingness_used_only_in_audit": True,
            "models_or_policies_updated": False,
            "thresholds_selected": False,
            "test_labels_may_not_be_used_for_tuning": True,
        },
    }
    stable_hash(report)  # Fail before creating output if a value is nonfinite.
    output = fresh_dir(out)
    report["out"] = str(output)
    write_json(output / "audit.json", report)
    return report
