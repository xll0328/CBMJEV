"""Held-out, group-aware task and one-step acquisition learning.

Models only receive partial hard semantic states. Complete cached responses and
labels are used by the *offline training target producer*, not model forward APIs.
"""

from itertools import combinations
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import candidate_actions


DEFAULTS = {
    "objective": "risk", "seed": 17, "device": "cpu", "hidden": 128,
    "head_epochs": 20, "policy_epochs": 20, "batch_size": 64,
    "learning_rate": 0.001, "weight_decay": 0.0, "dropout": 0.0,
    "masks_per_sample": 4, "include_pairs": True, "include_all": True,
    "max_pair_actions": 8, "actions_per_state": 64,
    "deterministic": True, "cpu_threads": 1, "class_weighting": "none",
}


def schema_signature(schema):
    """Semantic checkpoint identity, independent of sample provenance."""
    return schema.hash


def normalize_config(config, schema):
    cfg = {**DEFAULTS, **config}
    if cfg["objective"] not in {"risk", "value"}:
        raise ValueError("objective must be risk or value")
    if cfg["class_weighting"] not in ("none", "inverse_frequency"):
        raise ValueError("class_weighting must be none or inverse_frequency")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for name in ("hidden", "head_epochs", "policy_epochs", "batch_size", "masks_per_sample", "cpu_threads"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(name + " must be a positive integer")
    for name in ("max_pair_actions", "actions_per_state"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(name + " must be a nonnegative integer")
    for name in ("include_pairs", "include_all", "deterministic"):
        if type(cfg[name]) is not bool:
            raise ValueError(name + " must be boolean")
    if cfg["actions_per_state"] == 1:
        raise ValueError("actions_per_state must be zero (all) or at least two")
    for name in ("learning_rate", "weight_decay", "dropout"):
        if type(cfg[name]) not in (float, int) or not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError("invalid " + name)
    if cfg["learning_rate"] == 0 or cfg["dropout"] >= 1:
        raise ValueError("learning_rate must be positive and dropout below one")
    device = torch.device(cfg["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("supported training devices are cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    cfg["device"] = str(device)
    if "pairs" in cfg and cfg["pairs"] is not None:
        pairs = tuple(tuple(pair) for pair in cfg["pairs"])
    else:
        pairs = list(combinations(range(schema.num_groups), 2))
        # This fixed pool uses no responses/labels. It is NOT a learned
        # complementarity ranking. Persist the pool for identical online actions.
        rng = random.Random(cfg["seed"] + 811)
        rng.shuffle(pairs)
        pairs = tuple(sorted(pairs[:cfg["max_pair_actions"]]))
    if any(len(pair) != 2 or pair[0] >= pair[1]
           or any(type(g) is not int or not 0 <= g < schema.num_groups for g in pair)
           for pair in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("pairs must contain distinct ordered group-index pairs")
    cfg["pairs"] = [list(pair) for pair in pairs]
    return cfg


def training_rows(rows, split, schema):
    """Filter BEFORE inspecting responses or targets of any nontraining split."""
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError("nonempty " + split + " rows required")
    result = []
    seen = set()
    for row in selected:
        sample_id, group_id = row.get("sample_id"), row.get("group_id")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(group_id, str) or not group_id:
            raise ValueError("training rows need nonempty sample_id and group_id")
        if sample_id in seen:
            raise ValueError("duplicate training sample_id")
        seen.add(sample_id)
        z, y = row.get("z"), row.get("y")
        if not isinstance(z, (list, tuple)) or len(z) != schema.num_atoms:
            raise ValueError("training response width differs from schema")
        if any(type(value) is not int or not 0 <= value < count
               for value, count in zip(z, schema.num_categories)):
            raise ValueError("training responses must be complete hard runtime categories")
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid training target")
        result.append((tuple(z), y))
    return result


def validate_training_separation(rows):
    selected = [row for row in rows if row.get("split") in {"head_fit", "policy_fit"}]
    ownership, seen = {}, set()
    for row in selected:
        sid, gid, split = row.get("sample_id"), row.get("group_id"), row.get("split")
        if sid in seen:
            raise ValueError("sample_id overlaps head_fit/policy_fit")
        if gid in ownership and ownership[gid] != split:
            raise ValueError("group_id overlaps head_fit/policy_fit")
        seen.add(sid)
        ownership[gid] = split


def sample_group_masks(count, num_groups, rng):
    """Uniform cardinality then uniform subset; depends on RNG, not hidden z."""
    masks = []
    for _ in range(count):
        count_visible = rng.randrange(num_groups + 1)
        selected = set(rng.sample(range(num_groups), count_visible))
        masks.append(tuple(g in selected for g in range(num_groups)))
    return tuple(masks)


def mask_answers(answers, group_mask, schema):
    if len(answers) != schema.num_atoms or len(group_mask) != schema.num_groups:
        raise ValueError("state or group-mask width mismatch")
    observed = list(schema.empty_state())
    for g, visible in enumerate(group_mask):
        if visible:
            for atom in schema.groups[g].atoms:
                observed[atom] = answers[atom]
    return tuple(observed)


def validate_observed(observed, schema):
    if not isinstance(observed, (list, tuple)) or len(observed) != schema.num_atoms:
        raise ValueError("observed must be a fixed-width partial hard-state sequence")
    state = tuple(observed)
    if any(type(v) is not int or v < -1 or v >= count for v, count in zip(state, schema.num_categories)):
        raise ValueError("invalid observed hard category")
    for group in schema.groups:
        visibility = [state[a] >= 0 for a in group.atoms]
        if any(visibility) and not all(visibility):
            raise ValueError("partial query group: all atoms must be revealed together")
    return state


def validate_action(action, observed, schema):
    action = tuple(action)
    if (any(type(g) is not int or not 0 <= g < schema.num_groups for g in action)
            or tuple(sorted(set(action))) != action):
        raise ValueError("invalid acquisition action")
    if any(any(observed[a] >= 0 for a in schema.groups[g].atoms) for g in action):
        raise ValueError("action attempts to reacquire an observed group")
    return action


def encode_states(states, schema, device="cpu"):
    states = [validate_observed(state, schema) for state in states]
    if not states:
        return torch.empty((0, sum(schema.num_categories) + schema.num_atoms), device=device)
    z = torch.tensor(states, dtype=torch.long, device=device)
    mask = z >= 0
    safe = torch.where(mask, z, torch.zeros_like(z))
    pieces = [F.one_hot(safe[:, atom], count).float() * mask[:, atom:atom + 1]
              for atom, count in enumerate(schema.num_categories)]
    return torch.cat(pieces + [mask.float()], dim=-1)


def encode_actions(actions, schema, device="cpu"):
    # Actions are Python metadata: assemble once on host instead of dispatching
    # a separate CUDA indexing kernel (and index transfer) for every example.
    rows = [[0.0] * (schema.num_groups + 1) for _ in actions]
    for row, action in zip(rows, actions):
        for index in action or (-1,):
            row[index] = 1.0
    return torch.tensor(rows, dtype=torch.float32, device=device).reshape(
        len(actions), schema.num_groups + 1)


def _network(inputs, hidden, outputs, dropout):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, outputs))


class MaskedHead:
    def __init__(self, schema, config, class_weights=None):
        self.schema = schema
        self.config = dict(config)
        self.device = torch.device(config["device"])
        weighting = config.get("class_weighting", "none")
        if weighting == "none":
            if class_weights is not None:
                raise ValueError("unweighted head must not carry class weights")
            self.class_weights = None
            self._loss_weights = None
        elif weighting == "inverse_frequency":
            if (not isinstance(class_weights, (list, tuple)) or len(class_weights) != schema.num_classes
                    or any(type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0
                           for weight in class_weights)):
                raise ValueError("inverse_frequency head requires finite positive class weights for every class")
            self.class_weights = tuple(float(weight) for weight in class_weights)
            self._loss_weights = torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
            if not torch.isfinite(self._loss_weights).all() or not (self._loss_weights > 0).all():
                raise ValueError("class weights cannot be represented as finite positive float32")
        else:
            raise ValueError("class_weighting must be none or inverse_frequency")
        inputs = sum(schema.num_categories) + schema.num_atoms
        self.network = _network(inputs, config["hidden"], schema.num_classes, config["dropout"]).to(self.device)

    def cross_entropy(self, logits, targets, reduction="mean"):
        """Offline training loss; inference still only consumes observed states.

        Weighted mode averages per-example weighted CE, not PyTorch's weighted
        mean divided by the random minibatch sum of weights. Thus weights retain
        their meaning even at batch_size=1 and match per-example value targets.
        """
        if reduction not in ("mean", "none"):
            raise ValueError("head loss reduction must be mean or none")
        if self._loss_weights is None:
            # Preserve the exact previous default forward/backward numerical path.
            return F.cross_entropy(logits, targets, reduction=reduction)
        losses = F.cross_entropy(logits, targets, weight=self._loss_weights, reduction="none")
        return losses.mean() if reduction == "mean" else losses

    def probabilities_many(self, observed_states):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(encode_states(observed_states, self.schema, self.device))
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite task-head output")
            return tuple(tuple(row) for row in logits.softmax(-1).cpu().tolist())

    def probabilities(self, observed):
        return self.probabilities_many((observed,))[0]

    def predict(self, observed):
        probabilities = self.probabilities(observed)
        return max(range(len(probabilities)), key=probabilities.__getitem__)


class ActionController:
    def __init__(self, schema, config):
        self.schema = schema
        self.config = dict(config)
        self.objective = config["objective"]
        self.pairs = tuple(tuple(pair) for pair in config["pairs"])
        self.device = torch.device(config["device"])
        inputs = sum(schema.num_categories) + schema.num_atoms + schema.num_groups + 1
        self.network = _network(inputs, config["hidden"], 1, config["dropout"]).to(self.device)

    def predict(self, observed, actions):
        state = validate_observed(observed, self.schema)
        actions = tuple(validate_action(action, state, self.schema) for action in actions)
        if not actions:
            return ()
        self.network.eval()
        with torch.no_grad():
            encoded = encode_states((state,), self.schema, self.device).expand(len(actions), -1)
            x = torch.cat((encoded, encode_actions(actions, self.schema, self.device)), dim=-1)
            values = self.network(x).flatten()
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite acquisition-controller output")
            if self.objective == "risk":
                values = torch.sigmoid(values)
            scores = values.cpu().tolist()
        # The loss drop of taking no measurement is mathematically zero.
        return tuple(0.0 if self.objective == "value" and not action else float(score)
                     for action, score in zip(actions, scores))


def policy_examples(rows, schema, config, epoch=0):
    """Stream (before, action, after, y), keeping dense memory O(batch*D)."""
    rng = random.Random(config["seed"] + 104729 * (epoch + 1))
    order = list(range(len(rows)))
    rng.shuffle(order)
    for index in order:
        answers, y = rows[index]
        masks = list(sample_group_masks(config["masks_per_sample"], schema.num_groups, rng))
        masks[0] = (False,) * schema.num_groups
        if len(masks) > 1:
            masks[-1] = (True,) * schema.num_groups
        for mask in masks:
            before = mask_answers(answers, mask, schema)
            actions = list(candidate_actions(before, schema,
                                             include_pairs=config["include_pairs"],
                                             include_all=config["include_all"],
                                             pairs=tuple(tuple(pair) for pair in config["pairs"])))
            cap = config["actions_per_state"]
            if cap and len(actions) > cap:
                # Preserve STOP and all-remaining, sample other action IDs without
                # inspecting their answers. This is sampling, not oracle choice.
                nonempty = [action for action in actions if action]
                all_remaining = tuple(g for g in range(schema.num_groups) if not mask[g])
                retained = [()]
                if config["include_all"] and all_remaining in nonempty:
                    retained.append(all_remaining)
                    nonempty.remove(all_remaining)
                retained.extend(rng.sample(nonempty, cap - len(retained)))
                actions = retained
            for action in actions:
                after = list(before)
                for atom in schema.expand(action):
                    after[atom] = answers[atom]
                yield before, tuple(action), tuple(after), y


def _batches(iterable, size):
    buffer = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) == size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def action_targets(head, examples, objective):
    """Actual frozen-head outcomes using offline automatic response transitions."""
    before, _, after, ys = zip(*examples)
    y = torch.tensor(ys, dtype=torch.long, device=head.device)
    head.network.eval()
    with torch.no_grad():
        after_logits = head.network(encode_states(after, head.schema, head.device))
        if objective == "risk":
            target = after_logits.argmax(-1).ne(y).float()
        elif objective == "value":
            before_logits = head.network(encode_states(before, head.schema, head.device))
            target = head.cross_entropy(before_logits, y, reduction="none") - head.cross_entropy(after_logits, y, reduction="none")
        else:
            raise ValueError("unknown training objective")
    return target


def iter_risk_training_examples(rows, head, schema, config):
    """Public streaming targets for Nano-style outcome controllers.

    Only policy_fit is inspected. `error` is the observed frozen-f error after
    executing that action's cached automatic measurement and then stopping.
    The emitted model-facing example has no y, sample ID, or hidden responses.
    """
    if schema_signature(head.schema) != schema_signature(schema):
        raise ValueError("head and training schema differ")
    cfg = normalize_config({**config, "objective": "risk"}, schema)
    records = training_rows(rows, "policy_fit", schema)
    epoch = config.get("example_epoch", 0)
    if type(epoch) is not int or epoch < 0:
        raise ValueError("example_epoch must be a nonnegative integer")
    for examples in _batches(policy_examples(records, schema, cfg, epoch), cfg["batch_size"]):
        errors = action_targets(head, examples, "risk").cpu().tolist()
        for (before, action, _, _), error in zip(examples, errors):
            yield {"observed": before, "action": action, "error": float(error), "split": "policy_fit"}


def fit_models(rows, schema, config):
    cfg = normalize_config(config, schema)
    validate_training_separation(rows)
    head_rows = training_rows(rows, "head_fit", schema)
    policy_rows = training_rows(rows, "policy_fit", schema)
    class_counts = [0] * schema.num_classes
    for _, label in head_rows:
        class_counts[label] += 1
    class_weights = None
    if cfg["class_weighting"] == "inverse_frequency":
        if any(count == 0 for count in class_counts):
            raise ValueError("inverse_frequency requires every class in head_fit; no smoothing or held-out fallback")
        class_weights = tuple(len(head_rows) / (schema.num_classes * count) for count in class_counts)
    torch.manual_seed(cfg["seed"])
    if cfg["device"].startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(cfg["deterministic"])
    if cfg["device"] == "cpu":
        torch.set_num_threads(cfg["cpu_threads"])
    head = MaskedHead(schema, cfg, class_weights=class_weights)
    optimizer = torch.optim.AdamW(head.network.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = random.Random(cfg["seed"] + 31)
    head_losses = []
    for _ in range(cfg["head_epochs"]):
        head.network.train()
        order = list(range(len(head_rows)))
        rng.shuffle(order)
        loss_sum = count = 0
        for indices in _batches(order, cfg["batch_size"]):
            masks = sample_group_masks(len(indices), schema.num_groups, rng)
            states = [mask_answers(head_rows[index][0], mask, schema) for index, mask in zip(indices, masks)]
            targets = torch.tensor([head_rows[index][1] for index in indices], dtype=torch.long, device=head.device)
            loss = head.cross_entropy(head.network(encode_states(states, schema, head.device)), targets)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            count += len(indices)
        head_losses.append(loss_sum / count)
    head.network.eval().requires_grad_(False)
    # Separate initialization keeps the frozen head identical in risk/value runs
    # with the same head configuration and seed.
    torch.manual_seed(cfg["seed"] + 1009)
    controller = ActionController(schema, cfg)
    optimizer = torch.optim.AdamW(controller.network.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    policy_losses, example_counts = [], []
    for epoch in range(cfg["policy_epochs"]):
        controller.network.train()
        loss_sum = count = 0
        for examples in _batches(policy_examples(policy_rows, schema, cfg, epoch), cfg["batch_size"]):
            before, actions, _, _ = zip(*examples)
            target = action_targets(head, examples, cfg["objective"])
            x = torch.cat((encode_states(before, schema, controller.device),
                           encode_actions(actions, schema, controller.device)), dim=-1)
            logits = controller.network(x).flatten()
            loss = (F.binary_cross_entropy_with_logits(logits, target) if cfg["objective"] == "risk"
                    else F.mse_loss(logits, target))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite policy training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(examples)
            count += len(examples)
        policy_losses.append(loss_sum / count)
        example_counts.append(count)
    controller.network.eval().requires_grad_(False)
    report = {
        "training_protocol": "DISJOINT_HEAD_FIT_POLICY_FIT_NO_FINAL_REFIT",
        "evidence_status": "TRAINING_EXECUTED_REQUIRES_EXTERNAL_DATA_PROVENANCE",
        "objective": cfg["objective"], "seed": cfg["seed"], "device": cfg["device"],
        "torch_version": str(torch.__version__), "schema_signature": schema_signature(schema),
        "head_fit_rows": len(head_rows), "policy_fit_rows": len(policy_rows),
        "class_weighting": cfg["class_weighting"], "head_class_counts": class_counts,
        "head_class_weights": list(class_weights) if class_weights is not None else None,
        "class_weight_source": "head_fit_labels_only" if class_weights is not None else "not_used",
        "head_loss_definition": "mean_examples(w_y * CE)" if class_weights is not None else "mean_examples(CE)",
        "value_target_definition": "w_y * (CE_before - CE_after)" if class_weights is not None else "CE_before - CE_after",
        "risk_target_definition": "unweighted_indicator_of_post_action_error",
        "static_order_loss_definition": "unweighted_CE_independent_of_head_training_weighting",
        "num_atoms": schema.num_atoms, "num_groups": schema.num_groups,
        "head_loss": head_losses, "policy_loss": policy_losses,
        "policy_examples_per_epoch": example_counts, "max_dense_training_batch": cfg["batch_size"],
        "training_mask_distribution": "uniform group cardinality, uniform subset; policy adds empty/full endpoints",
        "pair_selection": "explicit frozen pool or seeded response-independent sample",
        "pair_candidates": cfg["pairs"], "heldout_labels_used": False,
        "calibration_guarantee": False,
        "limitations": ["one-step outcome model, not nonmyopic Bellman risk",
                        "no final refit or nested OOF implementation",
                        "cache provenance and responder supervised ancestry require external audit",
                        "no learned rollout-mask mixture in this version",
                        "minibatch-generated action examples are not independent sample-size evidence"],
    }
    return head, controller, report


def save_models(path, head, controller, report=None):
    path = Path(path)
    if path.exists():
        raise ValueError("checkpoint exists; refusing overwrite")
    if schema_signature(head.schema) != schema_signature(controller.schema):
        raise ValueError("head and controller schemas differ")
    if head.config.get("class_weighting", "none") != controller.config.get("class_weighting", "none"):
        raise ValueError("head and controller class-weighting configurations differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "cbmjev-learning-v1", "schema_signature": schema_signature(head.schema),
        "config": json.loads(json.dumps(controller.config)),
        "head_class_weights": list(head.class_weights) if head.class_weights is not None else None,
        "head": {key: value.detach().cpu() for key, value in head.network.state_dict().items()},
        "controller": {key: value.detach().cpu() for key, value in controller.network.state_dict().items()},
        "report": json.loads(json.dumps(report or {})),
    }
    torch.save(payload, path)


def load_models(path, schema, device="cpu"):
    # Only tensor dictionaries / primitive metadata. Never load pickled modules.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "cbmjev-learning-v1" or payload.get("schema_signature") != schema_signature(schema):
        raise ValueError("checkpoint format or semantic schema mismatch")
    cfg = normalize_config({**payload["config"], "device": device}, schema)
    head = MaskedHead(schema, cfg, class_weights=payload.get("head_class_weights"))
    controller = ActionController(schema, cfg)
    head.network.load_state_dict(payload["head"], strict=True)
    controller.network.load_state_dict(payload["controller"], strict=True)
    head.network.eval().requires_grad_(False)
    controller.network.eval().requires_grad_(False)
    return head, controller, payload.get("report", {})
