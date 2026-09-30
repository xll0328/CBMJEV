"""Prepare the native crowd-enVENT appraisal benchmark without model evaluation.

The generation ``emotion`` is the eliciting/prompted emotion, NOT a separately
collected post-event self-report. Author appraisals are separate human ratings.
Only generated text enters the model input; the original label-aware masked
text, reader labels, demographics, and author IDs never enter that input.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import zipfile

from .contracts import Schema, stable_hash
from .io import file_hash, fresh_dir, read_json, read_jsonl, write_json, write_jsonl


DATASET = "crowd_envent"
DEFAULT_SEED = 20260930
SOURCE_URL = "https://www.romanklinger.de/data-sets/crowd-enVent2023.zip"
PAPER_URL = "https://aclanthology.org/2023.cl-1.1/"
TARGET_SEMANTICS = "elicited_prompted_author_emotion"
EMOTIONS = ("joy", "sadness", "surprise", "anger", "fear", "disgust", "relief",
            "guilt", "shame", "trust", "pride", "boredom", "no-emotion")
# Definitions follow the release's generation questionnaire, including the
# direction of standards/social_norms (violation, not conformity).
APPRAISALS = (
    ("suddenness", "How sudden or abrupt was the event?"),
    ("familiarity", "How familiar was the event to the author?"),
    ("predict_event", "How predictable was the event's occurrence to the author?"),
    ("pleasantness", "How pleasant was the event for the author?"),
    ("unpleasantness", "How unpleasant was the event for the author?"),
    ("goal_relevance", "How important did the author expect the consequences to be?"),
    ("chance_responsblt", "How much was the event caused by chance, circumstances, or nature?"),
    ("self_responsblt", "How much did the author's own behavior cause the event?"),
    ("other_responsblt", "How much did another person's behavior cause the event?"),
    ("predict_conseq", "How much did the author anticipate the consequences?"),
    ("goal_support", "How positive did the author expect the consequences to be?"),
    ("urgency", "How much did the event require an immediate response?"),
    ("self_control", "How much could the author influence the ongoing event?"),
    ("other_control", "How much was another person influencing the ongoing event?"),
    ("chance_control", "How much did uncontrollable outside influences determine the situation?"),
    ("accept_conseq", "How easily did the author expect to live with unavoidable consequences?"),
    ("standards", "How much did the event conflict with the author's standards and ideals?"),
    ("social_norms", "How much did the actions violate laws or accepted social norms?"),
    ("attention", "How much attention did the situation demand from the author?"),
    ("not_consider", "How much did the author try to exclude the situation from their thoughts?"),
    ("effort", "How much energy did dealing with the situation require from the author?"),
)
APPRAISAL_FIELDS = tuple(cid for cid, _ in APPRAISALS)
ROLES = ("responder_fit", "head_fit", "policy_fit", "policy_tune", "confirmation")
FIT_ROLES = frozenset(("responder_fit", "head_fit", "policy_fit"))
# One fixed, case-insensitive lexicon for every row, independent of its label.
# The space spelling is an explicit orthographic variant of no-emotion.
MASK_TERMS = tuple(sorted(set(EMOTIONS + ("no emotion",)), key=lambda s: (-len(s), s)))
_MASK = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, MASK_TERMS)) + r")(?!\w)", re.I)
_PREFIX = re.compile(r"^\s*i\s+felt\s+\[EMOTION\]\s+(?:when|that|if)\b\s*", re.I)
GENERATION_FILE = "corpus/crowd-enVent_generation.tsv"
VALIDATION_FILE = "corpus/crowd-enVent_validation.tsv"


def appraisal_schema():
    """Standard Schema-compatible 21 x 5 concepts and the 13-class native task."""
    return {"schema_version": "cbmjev-schema-v1", "dataset": DATASET,
            "num_classes": len(EMOTIONS),
            "task": {"description": "Predict the native elicited/prompted author-emotion category from event text.",
                     "source_field": "generation.emotion", "semantics": TARGET_SEMANTICS,
                     "independent_post_event_self_report": False, "values": list(EMOTIONS)},
            "concepts": [{"id": cid,
                          "description": question + " Native author rating: 1 = Not at all; 5 = Extremely.",
                          "values": ["1", "2", "3", "4", "5"]}
                         for cid, question in APPRAISALS],
            "groups": [{"id": cid, "atoms": [i]} for i, cid in enumerate(APPRAISAL_FIELDS)]}


def normalize_text(text):
    """Fixed duplicate key: NFKC, case-folding, Unicode word tokens/whitespace."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def preprocess_text(text):
    """Mask every native emotion term, then remove a matching elicitation prefix.

    Natural-language synonyms are not exhaustively censored. This is a declared
    text-only rule, not the release's label-aware ``hidden_emo_text`` heuristic.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("generated_text must be nonempty")
    masked, count = _MASK.subn("[EMOTION]", unicodedata.normalize("NFKC", text))
    masked, prefix_count = _PREFIX.subn("", masked, count=1)
    masked = " ".join(masked.split())
    if not masked:
        raise ValueError("fixed text preprocessing produced empty input")
    return masked, count, prefix_count


def _read_source(source):
    source = Path(source)
    names = (GENERATION_FILE, VALIDATION_FILE, "readme.txt")
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            # Intentionally never read archive predictions or published outputs.
            contents = {name: archive.read(name) for name in names}
        source_files = [{"path": source.name, "sha256": file_hash(source), "bytes": source.stat().st_size}]
    else:
        contents = {name: (source / name).read_bytes() for name in names}
        source_files = [{"path": name, "sha256": file_hash(source / name),
                         "bytes": (source / name).stat().st_size} for name in names]
    tables = {}
    for name in (GENERATION_FILE, VALIDATION_FILE):
        reader = csv.DictReader(io.StringIO(contents[name].decode("utf-8-sig"), newline=""), delimiter="\t")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate source column names")
        tables[name] = (list(reader), fields)
    generation, fields = tables[GENERATION_FILE]
    validation, vfields = tables[VALIDATION_FILE]
    required = set(APPRAISAL_FIELDS) | {"text_id", "emotion", "generated_text"}
    if not required.issubset(fields) or "text_id" not in vfields:
        raise ValueError("source is missing required native crowd-enVENT fields")
    if not generation:
        raise ValueError("generation table is empty")
    ids = [row.get("text_id", "") for row in generation]
    if any(not isinstance(sid, str) or not sid.strip() for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("generation text_id must be nonempty and unique")
    if any(None in row or any(value is None for value in row.values()) for row in generation + validation):
        raise ValueError("source TSV row width differs from its header")
    test_ids = {row["text_id"] for row in validation}
    if not test_ids.issubset(set(ids)):
        raise ValueError("validation text_id has no matching generation record")
    metadata = {"source_url": SOURCE_URL, "paper_url": PAPER_URL, "source_files": source_files,
                "source_columns": fields, "validation_columns": vfields,
                "readme_sha256": hashlib.sha256(contents["readme.txt"]).hexdigest(),
                "generation_rows": len(generation), "validation_rows": len(validation),
                "official_test_rows": len(test_ids), "official_non_test_rows": len(ids) - len(test_ids)}
    return generation, test_ids, metadata


def _native_label(value, field):
    if value == "":
        return None, "MISSING_ANNOTATION"
    labels = EMOTIONS if field == "emotion" else ("1", "2", "3", "4", "5")
    if value not in labels:
        # Do not print raw row content or unknown labels into run logs.
        raise ValueError("unknown native category in " + field)
    return labels.index(value), "OBSERVED"


def _convert_and_group(rows, test_ids, metadata):
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    authors = [row.get("prolific_id", "").strip() for row in rows]
    has_author = "prolific_id" in metadata["source_columns"]
    if has_author and any(not author for author in authors):
        raise ValueError("published author column is partially missing; do not claim author isolation")
    seen = {"author": {}, "raw_text": {}, "model_text": {}}
    samples, masks, prefixes = [], [], []
    for index, row in enumerate(rows):
        text, masked_terms, prefix_count = preprocess_text(row["generated_text"])
        keys = {"raw_text": normalize_text(row["generated_text"]), "model_text": normalize_text(text)}
        if not all(keys.values()):
            raise ValueError("empty normalized text key")
        if has_author:
            keys["author"] = authors[index]
        for kind, key in keys.items():
            if key in seen[kind]:
                union(index, seen[kind][key])
            else:
                seen[kind][key] = index
        target, status = _native_label(row["emotion"], "emotion")
        concepts = []
        for field in APPRAISAL_FIELDS:
            value, annotation_status = _native_label(row[field], field)
            concepts.append({"concept_id": field, "value": value,
                             "annotation_status": annotation_status, "distribution": None})
        sid = row["text_id"]
        samples.append({"schema_version": "acam-data-v1", "dataset": DATASET,
                        "sample_id": DATASET + ":" + sid, "group_id": None,
                        "split": None, "fold_id": None,
                        "input": {"modality": "text", "text": text, "image_paths": []},
                        "target": {"value": target, "status": status}, "concepts": concepts,
                        "provenance": {"source_id": sid, "source_split": "official_test" if sid in test_ids else "official_non_test",
                                       "raw_sha256": stable_hash(row)},
                        "audit_metadata": {"target_semantics": TARGET_SEMANTICS,
                                           "author_hash": stable_hash(authors[index]) if has_author else None}})
        masks.append(masked_terms)
        prefixes.append(prefix_count)
    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    for indices in components.values():
        gid = DATASET + ":component:" + stable_hash(sorted(samples[i]["sample_id"] for i in indices))
        for i in indices:
            samples[i]["group_id"] = gid
    test = [i for i, row in enumerate(rows) if row["text_id"] in test_ids]
    rest = [i for i, row in enumerate(rows) if row["text_id"] not in test_ids]
    ta, ra = {authors[i] for i in test}, {authors[i] for i in rest}
    text_sets = [{normalize_text(rows[i]["generated_text"]) for i in part} for part in (test, rest)]
    metadata.update(
        author_field="prolific_id" if has_author else None,
        author_grouping="public_encrypted_author_id" if has_author else "unavailable_text_groups_only",
        unique_authors=len(set(authors)) if has_author else None,
        official_test_authors=len(ta) if has_author else None,
        official_non_test_authors=len(ra) if has_author else None,
        official_author_intersection=len(ta & ra) if has_author else None,
        official_test_rows_with_non_test_author=sum(authors[i] in ra for i in test) if has_author else None,
        official_normalized_text_intersection=len(text_sets[0] & text_sets[1]),
        normalized_text_duplicate_groups=sum(n > 1 for n in Counter(normalize_text(r["generated_text"]) for r in rows).values()),
        grouping_rule="author OR normalized original text OR normalized model input; transitive closure",
        text_normalization="Unicode NFKC, casefold, Unicode word tokens joined by one space",
        semantic_near_duplicates="NOT_CHECKED",
        components=len(components), largest_component=max(map(len, components.values())),
        target_field="generation.emotion", target_semantics=TARGET_SEMANTICS,
        target_values=list(EMOTIONS), target_missing=sum(s["target"]["status"] != "OBSERVED" for s in samples),
        appraisal_fields=list(APPRAISAL_FIELDS), appraisal_native_values=[1, 2, 3, 4, 5],
        appraisal_encoded_values=[0, 1, 2, 3, 4],
        appraisal_missing={field: sum(s["concepts"][i]["annotation_status"] != "OBSERVED" for s in samples)
                           for i, field in enumerate(APPRAISAL_FIELDS)},
        masking={"version": "native-category-terms-v1", "terms": list(MASK_TERMS),
                 "replacement": "[EMOTION]", "case_insensitive": True, "gold_label_used": False,
                 "original_hidden_emo_text_used": False, "rows_with_masked_terms": sum(n > 0 for n in masks),
                 "masked_term_occurrences": sum(masks), "elicitation_prefixes_removed": sum(prefixes),
                 "synonym_censoring": "not exhaustive; only frozen native category terms"})
    return sorted(samples, key=lambda sample: sample["sample_id"])


def inspect_crowd_envent(source):
    """Return only structural metadata; never sample IDs, row texts or row labels."""
    rows, test_ids, metadata = _read_source(source)
    _convert_and_group(rows, test_ids, metadata)
    return metadata


def _allocate(groups, fractions, seed, purpose):
    """One deterministic, size-first, class-balanced group allocation.

    Minimize the incremental squared discrepancy from each role's target total
    and per-class counts, normalized by that target. Labels only stratify; no
    predictions, learned values, or repeated split selection are involved.
    """
    if len(groups) < len(fractions):
        raise ValueError("not enough connected groups for the requested roles")
    total = sum(len(rows) for rows in groups.values())
    totals = Counter(s["target"]["value"] for rows in groups.values() for s in rows)
    sizes, classes, assignment = Counter(), {role: Counter() for role in fractions}, {}
    order = sorted(groups, key=lambda g: (-len(groups[g]), stable_hash([seed, purpose, g])))
    for index, group in enumerate(order):
        counts = Counter(s["target"]["value"] for s in groups[group])
        n = len(groups[group])
        empty = [role for role in fractions if sizes[role] == 0]
        choices = empty if len(order) - index == len(empty) else fractions

        def score(role):
            expected = total * fractions[role]
            delta = ((sizes[role] + n - expected) ** 2 - (sizes[role] - expected) ** 2) / max(expected, 1)
            for label, count in counts.items():
                expected_label = totals[label] * fractions[role]
                before = classes[role][label] - expected_label
                delta += ((before + count) ** 2 - before ** 2) / max(expected_label, 1)
            return delta, stable_hash([seed, purpose, group, role])

        role = min(choices, key=score)
        assignment[group] = role
        sizes[role] += n
        classes[role].update(counts)
    return assignment


def prepare_appraisal_data(source, out, seed=DEFAULT_SEED, source_revision="crowd-enVent2023"):
    """Freeze a new author/text-component split; refuse nonempty output paths."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output must be new or empty")
    rows, test_ids, metadata = _read_source(source)
    samples = _convert_and_group(rows, test_ids, metadata)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group_id"]].append(sample)
    assignments = _allocate(groups, {"responder_fit": .4, "head_fit": .2,
                                    "policy_development": .2, "confirmation": .2}, seed, "outer")
    policy = {g: rows for g, rows in groups.items() if assignments[g] == "policy_development"}
    assignments.update(_allocate(policy, {"policy_fit": .8, "policy_tune": .2}, seed, "policy"))
    members = []
    for sample in samples:
        role = assignments[sample["group_id"]]
        outer = "train" if role in FIT_ROLES else role
        sample["split"] = outer
        # Internal source/head tuning/OOF must be built inside their own role.
        sample["fold_id"] = None
        members.append({"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                        "split": role, "outer_split": outer})
    schema = appraisal_schema()
    Schema.from_dict(schema)
    report = {"schema_version": "cbmjev-appraisal-audit-v1", "dataset": DATASET,
              "status": "PREPARED_PROSPECTIVE_NO_MODEL_OUTCOMES", "seed": seed,
              "implementation_sha256": file_hash(Path(__file__)),
              "source_revision": source_revision, "schema_hash": stable_hash(schema),
              "split_hash": stable_hash(members), "sample_content_hash": stable_hash(samples),
              "adapter": metadata, "target_semantics": TARGET_SEMANTICS,
              "target_label_mapping": dict(enumerate(EMOTIONS)),
              "split_policy": {"version": "balanced-author-text-components-v1",
                               "source_head_policy_confirmation": [.4, .2, .2, .2],
                               "policy_fit_tune": [.8, .2], "same_confirmation_for_all_model_seeds": True,
                               "official_test_reused": False,
                               "reason": "prospective author/text isolation; original test has overlap",
                               "source_and_head_internal_splits": "must be constructed only within own role"},
              "role_counts": dict(Counter(m["split"] for m in members)),
              "role_group_counts": dict(Counter(assignments.values())),
              "rights": {"corpus_license": "UNSPECIFIED",
                         "research_use_basis": "official author-hosted release and README citation/use invitation; questionnaire states public anonymized research use",
                         "redistribution": "NOT_AUTHORIZED_BY_THIS_ADAPTER",
                         "code_MIT_is_not_corpus_license": True,
                         "article_license_is_not_assumed_to_cover_corpus": True},
              "checks": {"group_cross_role_overlap": 0, "known_native_categories": "CHECKED",
                         "target_derived_from_concepts": False, "concepts_derived_from_target": False,
                         "model_outcomes_accessed": False, "named_corpus_license": "NOT_ESTABLISHED"}}
    # Verify actual membership, not merely intended assignment.
    observed = defaultdict(set)
    for member in members:
        observed[member["group_id"]].add(member["split"])
    if any(len(roles) != 1 for roles in observed.values()):
        raise AssertionError("group crosses roles")
    fresh_dir(out)
    paths = {name: out / (name + suffix) for name, suffix in
             (("schema", ".json"), ("samples", ".jsonl"), ("membership", ".jsonl"), ("audit", ".json"))}
    write_json(paths["schema"], schema)
    write_jsonl(paths["samples"], samples)
    write_jsonl(paths["membership"], members)
    write_json(paths["audit"], report)
    paths["report"] = report
    return paths


def load_role_records(prepared, roles, allow_confirmation=False):
    """Load explicitly requested roles, validating immutable content and grouping.

    Confirmation is denied by default. This is a protocol guard, not an OS-level
    security boundary; source fitting must request only ``responder_fit``.
    """
    if isinstance(roles, str):
        roles = (roles,)
    requested = set(roles)
    if not requested or requested - set(ROLES):
        raise ValueError("request explicit known appraisal roles")
    if "confirmation" in requested and not allow_confirmation:
        raise ValueError("confirmation is locked; explicitly frozen evaluation required")
    prepared = Path(prepared)
    audit = read_json(prepared / "audit.json")
    schema = read_json(prepared / "schema.json")
    members = read_jsonl(prepared / "membership.jsonl")
    samples = read_jsonl(prepared / "samples.jsonl")
    if stable_hash(schema) != audit["schema_hash"] or stable_hash(members) != audit["split_hash"]:
        raise ValueError("prepared schema or membership hash changed")
    if stable_hash(samples) != audit["sample_content_hash"]:
        raise ValueError("prepared sample contents changed")
    Schema.from_dict(schema)
    mapping = {m["sample_id"]: m for m in members}
    if len(mapping) != len(members) or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("duplicate prepared sample IDs")
    if set(mapping) != {s["sample_id"] for s in samples}:
        raise ValueError("samples and membership IDs differ")
    grouped, selected = {}, []
    for sample in samples:
        member = mapping[sample["sample_id"]]
        role = member["split"]
        expected_outer = "train" if role in FIT_ROLES else role
        if role not in ROLES or sample["group_id"] != member["group_id"] or sample["split"] != expected_outer or member["outer_split"] != expected_outer:
            raise ValueError("invalid sample role/group join")
        group = member["group_id"]
        if group in grouped and grouped[group] != role:
            raise ValueError("component spans roles")
        grouped[group] = role
        if role in requested:
            selected.append(sample)
    if not selected:
        raise ValueError("requested roles contain no samples")
    return selected
