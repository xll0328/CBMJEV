"""Lossless streaming risk decision sets, not a new objective or Q target.

Inputs must remain immutable throughout validation and iteration (including
JSONL files). Full package/provenance validation and a complete structural pass
finish before the first set can reach a consumer. Memory holds one occurrence's
actions, never a global sample/state aggregation. The coverage preflight also
retains an O(number of samples) identity set, not an O(number of events) buffer.
"""
from dataclasses import dataclass

from .contracts import candidate_actions
from .crossfit_training import _validate_target_package
from .learning import normalize_config


@dataclass(frozen=True)
class DecisionInputs:
    observed: tuple
    actions: tuple


@dataclass(frozen=True)
class RiskSupervision:
    risk_targets: tuple


@dataclass(frozen=True)
class DecisionMetadata:
    source_artifact_id: str
    source_package_sha256: str
    occurrence: int
    epoch: int
    sample_id: str
    group_id: str


@dataclass(frozen=True)
class DecisionSet:
    model_inputs: DecisionInputs
    supervision: RiskSupervision
    metadata: DecisionMetadata


def _sets(package, schema, cfg):
    current, occurrence = [], 0

    def finish(records, index):
        first = records[0]
        identity = (first["epoch"], first["sample_id"], first["group_id"],
                    tuple(first["observed"]))
        if any((r["epoch"], r["sample_id"], r["group_id"], tuple(r["observed"])) != identity
               for r in records):
            raise ValueError("decision set identity/state changes without STOP boundary")
        actions = tuple(tuple(r["action"]) for r in records)
        if actions[0] != () or len(set(actions)) != len(actions):
            raise ValueError("decision set requires initial STOP and unique actions")
        expected_order = candidate_actions(identity[3], schema,
            include_pairs=cfg["include_pairs"], include_all=cfg["include_all"], pairs=cfg["pairs"])
        expected = set(expected_order)
        cap = cfg["actions_per_state"]
        if cap and cap < len(expected):
            raise ValueError("complete decision sets required: candidate cap removed actions")
        count = len(expected)
        if len(actions) != count or set(actions) != expected:
            raise ValueError("decision set candidate count/pool mismatch")
        if actions != expected_order:
            raise ValueError("decision set candidate order differs from producer")
        return DecisionSet(DecisionInputs(identity[3], actions),
            RiskSupervision(tuple(r["target"] for r in records)),
            DecisionMetadata(package["target_artifact_id"], package["package_sha256"],
                             index, identity[0], identity[1], identity[2]))

    # An occurrence starts at every STOP, including adjacent full-visible STOPs.
    # Bound the buffer even for a malformed stream lacking later boundaries.
    maximum = 2 + schema.num_groups + (len(cfg["pairs"]) if cfg["include_pairs"] else 0)
    if cfg["actions_per_state"]:
        maximum = min(maximum, cfg["actions_per_state"])
    for record in package["records"]:
        if not record["action"]:
            if current:
                yield finish(current, occurrence)
                occurrence += 1
            current = [record]
        else:
            if not current:
                raise ValueError("decision stream must start with STOP")
            current.append(record)
            if len(current) > maximum:
                raise ValueError("decision set exceeds bounded candidate count")
    if current:
        yield finish(current, occurrence)


def _check_occurrences(sets, package, cfg):
    """Verify the generator's epoch/sample blocks without storing all sets."""
    expected_samples = set(package["target_sample_ids"])
    epoch, sample, count, seen = 0, None, 0, set()
    for decision in sets:
        meta = decision.metadata
        if (meta.epoch, meta.sample_id) != (epoch, sample):
            if sample is not None and count != cfg["masks_per_sample"]:
                raise ValueError("decision occurrence count differs from masks_per_sample")
            if meta.epoch != epoch:
                if meta.epoch != epoch + 1 or seen != expected_samples:
                    raise ValueError("decision epoch order/sample coverage mismatch")
                epoch, seen = meta.epoch, set()
            if meta.sample_id in seen:
                raise ValueError("decision sample occurrences are not contiguous")
            seen.add(meta.sample_id)
            sample, count = meta.sample_id, 0
        count += 1
    if (count != cfg["masks_per_sample"] or seen != expected_samples
            or epoch != cfg["policy_epochs"] - 1):
        raise ValueError("decision occurrence/epoch coverage incomplete")


def iter_decision_sets(package, schema, config):
    """Return validated occurrences; forward only each ``model_inputs`` field.

    Supervision retains the original scalar risks verbatim. No argmin/CE labels,
    continuation values, or unrevealed concept targets are invented here.
    """
    cfg = normalize_config(config, schema)
    if cfg["objective"] != "risk":
        raise ValueError("decision-set adapter requires original risk targets")
    _validate_target_package(package, schema, cfg)
    _check_occurrences(_sets(package, schema, cfg), package, cfg)
    return _sets(package, schema, cfg)
