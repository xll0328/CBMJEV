"""Group-aware sequential inference, with separate live and offline environments."""
import math
import hashlib
from pathlib import Path
import random
import time

from .contracts import DeclaredCost, ModelInput, candidate_actions, stable_hash


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def load_payload(input_record, raw_root=None, *, png_compress_level=0):
    """Resolve filenames outside the responder; lossless, fast in-memory PNG.

    Compression changes byte digests, not RGB pixels. Frozen older releases
    retain their original compression and cache identities. No disk PNG cache
    is created; level six is available for equivalence diagnostics only.
    """
    if type(png_compress_level) is not int or not 0 <= png_compress_level <= 9:
        raise ValueError("png_compress_level must be an integer in [0,9]")
    if input_record["modality"] == "text":
        return ModelInput(text=input_record["text"])
    if input_record["modality"] != "image" or raw_root is None:
        raise ValueError("image inputs require a local raw_root")
    import io
    from PIL import Image, ImageOps
    root = Path(raw_root).resolve()
    images = []
    for relative in input_record["image_paths"]:
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("image path must be relative and cannot traverse parents")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("image symlink escapes raw_root") from exc
        with Image.open(path) as original:
            pixels = ImageOps.exif_transpose(original).convert("RGB")
            buffer = io.BytesIO()
            pixels.save(buffer, format="PNG", compress_level=png_compress_level)
            images.append(buffer.getvalue())
    return ModelInput(images=tuple(images))


class ReplayEnvironment:
    mode = "offline_replay"

    def __init__(self, answers, schema):
        schema.validate_state(answers, complete=True)
        self._answers = tuple(answers)
        self.schema = schema
        self._state = schema.empty_state()
        self.calls = 0
        self.preprocess_ms = 0.0
        self.responder_ms = 0.0
        self.backend_stats = []

    def state(self):
        return self._state

    def query(self, action):
        atoms = self.schema.expand(action)
        values = tuple(self._answers[a] for a in atoms)
        self._state = self.schema.reveal(self._state, action, values)
        self.calls += 1
        return values


class LiveEnvironment:
    mode = "live"

    def __init__(self, input_record, schema, responder, raw_root=None, device="cpu", expected_digest=None,
                 expected_responses=None):
        # Never pass the complete canonical record (which includes gold c/y).
        self._input = input_record
        self.schema, self._responder = schema, responder
        self._raw_root, self._device = raw_root, device
        self._expected_digest = expected_digest
        if expected_responses is not None:
            schema.validate_state(expected_responses, complete=True)
        self._expected_responses = None if expected_responses is None else tuple(expected_responses)
        self._state = schema.empty_state()
        self._session = None
        self._payload = None
        self.calls = 0
        self.preprocess_ms = 0.0
        self.responder_ms = 0.0
        self.backend_stats = []

    def state(self):
        return self._state

    def query(self, action):
        atoms = self.schema.expand(action)
        if not atoms:
            raise ValueError("STOP must not invoke a responder")
        if any(self.schema.group_mask(self._state)[g] for g in action):
            raise ValueError("a query group cannot be acquired twice")
        if self._payload is None:
            begin = time.perf_counter()
            self._payload = load_payload(self._input, self._raw_root)
            if self._expected_digest is not None:
                digest = stable_hash({"text": self._payload.text,
                    "images": [hashlib.sha256(b).hexdigest() for b in self._payload.images]})
                if digest != self._expected_digest:
                    raise ValueError("live input content differs from the frozen response cache")
            self.preprocess_ms += (time.perf_counter() - begin) * 1000
        synchronize(self._device)
        begin = time.perf_counter()
        if self._session is None and hasattr(self._responder, "start_session"):
            self._session = self._responder.start_session(self._payload)
        if self._session is not None:
            values = self._session.respond(atoms)
            stats = getattr(self._session, "last_stats", {})
        else:
            values = self._responder.respond(self._payload, atoms)
            stats = getattr(self._responder, "last_stats", {})
        synchronize(self._device)
        self.responder_ms += (time.perf_counter() - begin) * 1000
        self.calls += 1
        # Statistics remain external to semantic observations and model inputs.
        self.backend_stats.append(dict(stats))
        self._state = self.schema.reveal(self._state, action, tuple(values))
        if self._expected_responses is not None and tuple(values) != tuple(self._expected_responses[a] for a in atoms):
            raise ValueError("live acquired answers differ from the training cache; frozen flat-replay equivalence failed")
        return tuple(values)


def choose_action(observed, actions, *, method, controller=None, order=(),
                  cost=None, cost_weight=0.0, remaining_budget=math.inf, remaining_groups=None):
    """No raw x, IDs, hidden responses, labels or actual timing in this API."""
    cost = cost or DeclaredCost()
    if method == "structured_choice":
        if controller is None or getattr(controller, "objective", None) != "choice":
            raise ValueError("structured_choice requires an explicit Choice controller")
        scores = tuple(float(v) for v in controller.predict_logits(observed, actions,
            remaining_groups=remaining_groups, cost=cost, cost_weight=cost_weight))
        if len(scores) != len(actions) or not scores or any(not math.isfinite(v) for v in scores):
            raise ValueError("Choice controller returned invalid logits")
        # Cost is already a conditioning input to the trained Choice objective.
        # Do not reinterpret preference logits as risks or penalize them twice.
        best = min(range(len(actions)), key=lambda i: (
            -scores[i], bool(actions[i]), cost(observed, actions[i]), actions[i]))
        return actions[best], {",".join(map(str, a)) if a else "STOP": v
                              for a, v in zip(actions, scores)}
    if method == "learned_static_mask":
        if controller is None:
            raise ValueError("learned_static_mask requires fitted global mask")
        selected = controller.selected_groups()
        seen = controller.schema.group_mask(observed)
        action = () if all(seen[g] for g in selected) else selected
        if action not in actions:
            raise ValueError("learned static subset is infeasible; never truncate")
        return action, {}
    if method == "brig":
        if controller is None or type(remaining_groups) is not int:
            raise ValueError("BRiG requires a controller and integer remaining group budget")
        action = controller.choose(observed, remaining_groups)
        if action not in actions or (remaining_groups > 0 and len(action) != 1):
            raise ValueError("BRiG returned an illegal fixed-budget singleton action")
        return action, {}
    if method == "stop" or len(actions) == 1:
        return (), {}
    if method == "all":
        # run_episode validates that one full initial batch is feasible. Never
        # silently relabel a budgeted largest-batch heuristic as all-at-once.
        return min(actions, key=lambda a: (-len(a), cost(observed, a), a)), {}
    if method in ("fixed", "random", "static"):
        for group in order:
            if (group,) in actions:
                return (group,), {}
        return (), {}
    if method == "lookahead":
        if controller is None:
            raise ValueError("lookahead requires a fitted empirical policy")
        action = controller.choose(observed, actions, remaining_budget,
                                   cost, cost_weight=cost_weight, remaining_groups=remaining_groups)
        if action not in actions:
            raise ValueError("lookahead returned an illegal action")
        return action, {}
    if method not in ("risk", "value", "static_value", "value_singleton", "nano_risk", "nano_static_risk", "matched_mlp_risk") or controller is None:
        raise ValueError("unknown method or missing controller: " + method)
    objective = getattr(controller, "objective", "risk")
    if (method in ("value", "static_value", "value_singleton")) != (objective == "value"):
        raise ValueError("controller objective does not match evaluation method")
    scores = tuple(float(v) for v in controller.predict(observed, actions))
    if len(scores) != len(actions) or any(not math.isfinite(v) for v in scores):
        raise ValueError("controller returned invalid action scores")
    if objective == "risk" and any(v < 0 or v > 1 for v in scores):
        raise ValueError("risk scores must be independent probabilities in [0, 1]")
    objectives = [(-v if objective == "value" else v) + cost_weight * cost(observed, a)
                  for a, v in zip(actions, scores)]
    # STOP wins exact ties, then cheaper action, then lexicographic IDs.
    best = min(range(len(actions)), key=lambda i: (
        objectives[i], bool(actions[i]), cost(observed, actions[i]), actions[i]))
    return actions[best], {",".join(map(str, a)) if a else "STOP": v
                           for a, v in zip(actions, scores)}


def run_episode(env, schema, head, *, method="risk", controller=None,
                cost=None, cost_weight=0.03, max_groups=None, max_cost=None,
                include_pairs=True, include_all=True, pairs=None, order=None,
                seed=17, device="cpu"):
    """Return an unlabeled transcript. The caller attaches evaluator-only metadata."""
    cost = cost or DeclaredCost()
    max_groups = schema.num_groups if max_groups is None else max_groups
    if type(max_groups) is not int or not 0 <= max_groups <= schema.num_groups:
        raise ValueError("max_groups must be in [0, K]")
    if max_cost is not None and (not math.isfinite(max_cost) or max_cost < 0):
        raise ValueError("max_cost must be finite and nonnegative")
    if not math.isfinite(cost_weight) or cost_weight < 0:
        raise ValueError("cost_weight must be finite and nonnegative")
    if method == "brig" and max_cost is not None:
        raise ValueError("BRiG supports fixed group budgets only, not max_cost constraints")
    if method == "learned_static_mask":
        if controller is None or env.state() != schema.empty_state():
            raise ValueError("learned static mask requires a controller and fresh empty history")
        selected = controller.selected_groups()
        if len(selected) > max_groups:
            raise ValueError("learned static mask K exceeds evaluation budget; cannot truncate")
        if max_cost is not None and cost(schema.empty_state(), selected) > max_cost + 1e-10:
            raise ValueError("learned static mask exceeds declared cost budget")
    if method == "all":
        full = tuple(range(schema.num_groups))
        if not include_all or max_groups != schema.num_groups:
            raise ValueError("all means one full initial batch: include_all=True and max_groups=K required")
        if max_cost is not None and cost(schema.empty_state(), full) > max_cost + 1e-10:
            raise ValueError("full all-at-once is infeasible under max_cost; run it as a separate full-budget reference")
        if env.state() != schema.empty_state():
            raise ValueError("all-at-once requires a fresh empty history")
    order = tuple(range(schema.num_groups)) if order is None else tuple(order)
    if sorted(order) != list(range(schema.num_groups)):
        raise ValueError("static order must be a permutation of query groups")
    if method == "random":
        temporary = list(range(schema.num_groups))
        random.Random(seed).shuffle(temporary)
        order = tuple(temporary)  # No sample ID, batch position or raw input seed.
    synchronize(device)
    started = time.perf_counter()
    declared, policy_ms, steps = 0.0, 0.0, []
    for _ in range(schema.num_groups + 1):
        observed = env.state()
        synchronize(device)
        begin = time.perf_counter()
        actions = candidate_actions(observed, schema, include_pairs, include_all, pairs)
        if method == "learned_static_mask":
            selected = controller.selected_groups()
            actions = ((), selected) if selected and observed == schema.empty_state() else ((),)
        if method in ("static_value", "nano_static_risk"):
            # Select the next prefix item BEFORE feasibility filtering: an
            # unaffordable next item must not silently reorder the static list.
            next_group = next((g for g in order if (g,) in actions), None)
            actions = tuple(a for a in actions if not a or a == (next_group,))
        elif method in ("value_singleton", "brig", "structured_choice"):
            actions = tuple(a for a in actions if len(a) <= 1)
        acquired = sum(schema.group_mask(observed))
        remaining = math.inf if max_cost is None else max(0.0, max_cost - declared)
        actions = tuple(a for a in actions if len(a) + acquired <= max_groups and
                        cost(observed, a) <= remaining + 1e-10)
        action, scores = choose_action(observed, actions, method=method,
                                      controller=controller, order=order, cost=cost,
                                      cost_weight=cost_weight, remaining_budget=remaining,
                                      remaining_groups=max_groups - acquired)
        synchronize(device)
        policy_ms += (time.perf_counter() - begin) * 1000
        step = {"before": list(observed), "action": list(action), "scores": scores,
                "legal_candidate_count": len(actions), "scored_candidate_count": len(scores),
                "declared_cost": cost(observed, action)}
        if not action:
            step["decision"] = "STOP"
            steps.append(step)
            break
        values = env.query(action)
        declared += step["declared_cost"]
        step.update(decision="ACQUIRE", atom_ids=list(schema.expand(action)), values=list(values))
        steps.append(step)
    else:
        raise RuntimeError("episode did not terminate within K+1 steps")
    synchronize(device)
    begin = time.perf_counter()
    probabilities = tuple(float(p) for p in head.probabilities(env.state()))
    if len(probabilities) != schema.num_classes or any(
            not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("task head returned invalid probabilities")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError("task probabilities must sum to one")
    prediction = max(range(len(probabilities)), key=lambda i: probabilities[i])
    synchronize(device)
    head_ms = (time.perf_counter() - begin) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    result = {"method": method, "prediction": prediction,
              "probabilities": list(probabilities), "final_state": list(env.state()),
              "queried_groups": [i for i, present in enumerate(schema.group_mask(env.state())) if present],
              "queried_atoms": [i for i, v in enumerate(env.state()) if v != -1],
              "calls": env.calls, "declared_cost": declared, "cost_units": cost.units,
              "mode": env.mode, "steps": steps,
              "backend_stats": env.backend_stats}
    if env.mode == "live":
        result.update(total_wall_ms=total_ms, preprocess_ms=env.preprocess_ms,
                      responder_ms=env.responder_ms, policy_ms=policy_ms, head_ms=head_ms,
                      coordinator_ms=max(0.0, total_ms - env.preprocess_ms - env.responder_ms - policy_ms - head_ms),
                      timing_scope="warm_loaded_models_single_case_serial_includes_lazy_preprocessing",
                      evidence_status="LIVE_MEASUREMENT_NOT_YET_REPLICATED")
    else:
        result.update(replay_wall_ms=total_ms, evidence_status="OFFLINE_REPLAY_NOT_LATENCY")
    return result
