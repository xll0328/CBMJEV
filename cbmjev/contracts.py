"""Small immutable model-facing contracts. Metadata belongs to the orchestrator.

Hard observations use semantic category integers, followed by two explicit
runtime statuses: UNCERTAIN = C and NOT_APPLICABLE = C + 1. Unqueried = -1.
Gold annotation missingness never determines a runtime status.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Optional, Tuple


ROLES = ("responder_fit", "head_fit", "policy_fit", "validation", "calibration", "test")


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Concept:
    id: str
    description: str
    values: Tuple[str, ...]


@dataclass(frozen=True)
class QueryGroup:
    id: str
    atoms: Tuple[int, ...]


@dataclass(frozen=True)
class Schema:
    dataset: str
    num_classes: int
    concepts: Tuple[Concept, ...]
    groups: Tuple[QueryGroup, ...]

    def __post_init__(self):
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a nonempty string")
        if type(self.num_classes) is not int or self.num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if not self.concepts or not self.groups:
            raise ValueError("schema requires concepts and query groups")
        for entries in (self.concepts, self.groups):
            ids = [entry.id for entry in entries]
            if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("concept/group IDs must be unique nonempty strings")
        for concept in self.concepts:
            if not concept.description or len(concept.values) < 2:
                raise ValueError("each concept needs a description and >= 2 values")
            if any(not isinstance(v, str) or not v for v in concept.values):
                raise ValueError("semantic value names must be nonempty strings")
            if len(set(concept.values)) != len(concept.values):
                raise ValueError("semantic value names must be unique per concept")
        covered = []
        for group in self.groups:
            if not group.atoms or any(type(a) is not int for a in group.atoms):
                raise ValueError("group atoms must be nonempty integer indices")
            if tuple(sorted(set(group.atoms))) != group.atoms:
                raise ValueError("group atoms must be sorted and unique")
            covered.extend(group.atoms)
        if sorted(covered) != list(range(self.num_atoms)):
            raise ValueError("query groups must partition all atoms exactly once")

    @property
    def num_atoms(self):
        return len(self.concepts)

    @property
    def num_groups(self):
        return len(self.groups)

    @property
    def value_counts(self):
        return tuple(len(c.values) for c in self.concepts)

    @property
    def num_categories(self):
        return tuple(n + 2 for n in self.value_counts)

    @property
    def hash(self):
        return stable_hash(self.to_dict())

    def to_dict(self):
        return {"schema_version": "cbmjev-schema-v1", "dataset": self.dataset,
                "num_classes": self.num_classes,
                "concepts": [{"id": c.id, "description": c.description,
                              "values": list(c.values)} for c in self.concepts],
                "groups": [{"id": g.id, "atoms": list(g.atoms)} for g in self.groups]}

    @classmethod
    def from_dict(cls, obj):
        if obj.get("schema_version") != "cbmjev-schema-v1":
            raise ValueError("expected cbmjev-schema-v1; use prepare to convert canonical data")
        return cls(obj["dataset"], obj["num_classes"],
                   tuple(Concept(c["id"], c["description"], tuple(c["values"]))
                         for c in obj["concepts"]),
                   tuple(QueryGroup(g["id"], tuple(g["atoms"])) for g in obj["groups"]))

    def empty_state(self):
        return (-1,) * self.num_atoms

    def validate_state(self, observed, complete=False):
        if len(observed) != self.num_atoms:
            raise ValueError("observation width does not match schema")
        for value, size in zip(observed, self.num_categories):
            if type(value) is not int or value < (0 if complete else -1) or value >= size:
                raise ValueError("invalid hard category or runtime status")
        for group in self.groups:
            present = [observed[a] != -1 for a in group.atoms]
            if any(present) and not all(present):
                raise ValueError("partial query group: a group must reveal all its atoms")

    def group_mask(self, observed):
        self.validate_state(observed)
        return tuple(observed[g.atoms[0]] != -1 for g in self.groups)

    def expand(self, action):
        if not isinstance(action, tuple):
            action = tuple(action)
        if any(type(g) is not int or g < 0 or g >= self.num_groups for g in action):
            raise ValueError("invalid query group index")
        if tuple(sorted(set(action))) != action:
            raise ValueError("actions must contain sorted, unique group indices")
        return tuple(sorted(a for g in action for a in self.groups[g].atoms))

    def reveal(self, observed, action, values):
        mask = self.group_mask(observed)
        atoms = self.expand(action)
        if any(mask[g] for g in action):
            raise ValueError("a query group cannot be acquired twice")
        if len(values) != len(atoms):
            raise ValueError("responder returned the wrong number of atoms")
        state = list(observed)
        for atom, value in zip(atoms, values):
            state[atom] = value
        result = tuple(state)
        self.validate_state(result)
        return result


@dataclass(frozen=True)
class ModelInput:
    """Content only: no filenames, IDs, labels, split, or cost metadata."""
    text: Optional[str] = None
    images: Tuple[bytes, ...] = ()

    def __post_init__(self):
        if (self.text is None) == (not self.images):
            raise ValueError("provide exactly one modality: text or image bytes")
        if self.text is not None and (not isinstance(self.text, str) or not self.text.strip()):
            raise ValueError("text input cannot be empty")
        if any(not isinstance(b, bytes) or not b for b in self.images):
            raise ValueError("images must contain nonempty bytes")


@dataclass(frozen=True)
class DeclaredCost:
    """Frozen history-only cost proxy; never consumes measured case-specific time."""
    setup: float = 0.0
    call: float = 0.0
    per_group: float = 1.0
    units: str = "declared_query_units"

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or x < 0 for x in (self.setup, self.call, self.per_group)):
            raise ValueError("declared costs must be finite nonnegative numbers")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("cost units must be declared")

    def __call__(self, observed, action):
        if not action:
            return 0.0
        return self.call + self.per_group * len(action) + (
            self.setup if all(v == -1 for v in observed) else 0.0)

    def to_dict(self):
        return {"setup": self.setup, "call": self.call,
                "per_group": self.per_group, "units": self.units}


def candidate_actions(observed, schema, include_pairs=True, include_all=True, pairs=None):
    """STOP first. Construction uses acquired identities, not unobserved values."""
    mask = schema.group_mask(observed)
    remaining = tuple(g for g, present in enumerate(mask) if not present)
    actions = [()] + [(g,) for g in remaining]
    if include_pairs:
        selected = itertools.combinations(remaining, 2) if pairs is None else pairs
        for pair in selected:
            pair = tuple(pair)
            schema.expand(pair)
            if len(pair) != 2:
                raise ValueError("pair candidates must have length two")
            if all(g in remaining for g in pair):
                actions.append(pair)
    if include_all and remaining:
        actions.append(remaining)
    return tuple(dict.fromkeys(actions))


def validate_rows(rows, schema, required_roles=()):
    """Strict replay input checks; provenance is separately audited, not inferred."""
    ids, groups, counts = set(), {}, {}
    for row in rows:
        sid, gid, role = row.get("sample_id"), row.get("group_id"), row.get("split")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("sample_id must be a unique nonempty string")
        if not isinstance(gid, str) or not gid or role not in ROLES:
            raise ValueError("invalid group_id or split role")
        if gid in groups and groups[gid] != role:
            raise ValueError("group leakage across split roles")
        ids.add(sid)
        groups[gid] = role
        counts[role] = counts.get(role, 0) + 1
        schema.validate_state(row.get("z", ()), complete=True)
        if type(row.get("y")) is not int or not 0 <= row["y"] < schema.num_classes:
            raise ValueError("target must be an integer within the declared class vocabulary")
    if any(not counts.get(role) for role in required_roles):
        raise ValueError("missing required roles: " + ", ".join(
            r for r in required_roles if not counts.get(r)))
    return {"rows": len(rows), "groups": len(groups), "roles": counts}
