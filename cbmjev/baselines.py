"""Training-only static acquisition and a small empirical model-based baseline.

The lookahead baseline is NOT an ACO or BRiG reproduction, and its exactness is
only relative to its finite empirical response/label distribution and action set.
"""

from collections import Counter
import math
import random

from .contracts import candidate_actions
from .learning import _batches, mask_answers, training_rows, validate_action, validate_observed


def _mean_ce(head, states_and_labels, batch_size):
    total = count = 0
    for batch in _batches(states_and_labels, batch_size):
        states, labels = zip(*batch)
        for probabilities, y in zip(head.probabilities_many(states), labels):
            total += -math.log(max(float(probabilities[y]), 1e-12))
            count += 1
    if not count:
        raise ValueError("cannot estimate loss from zero examples")
    return total / count


def fit_static_order(rows, head, schema, config=None):
    """Greedy global-prefix order; never select an action per training sample.

    At each prefix, average actual signed loss changes across policy_fit before
    selecting the single global next group. Labels of validation/test are ignored.
    """
    config = dict(config or {})
    records = training_rows(rows, "policy_fit", schema)
    max_rows = config.get("static_max_rows", 0)
    batch_size = config.get("batch_size", 64)
    if type(max_rows) is not int or max_rows < 0 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("invalid static-order sample or batch limit")
    if max_rows and len(records) > max_rows:
        records = random.Random(config.get("seed", 17)).sample(records, max_rows)
    order, stages = [], []
    current_mask = [False] * schema.num_groups
    current_loss = _mean_ce(head, ((mask_answers(z, current_mask, schema), y) for z, y in records), batch_size)
    while len(order) < schema.num_groups:
        scores = {}
        losses = {}
        for group in range(schema.num_groups):
            if current_mask[group]:
                continue
            after_mask = list(current_mask)
            after_mask[group] = True
            after_loss = _mean_ce(head, ((mask_answers(z, after_mask, schema), y) for z, y in records), batch_size)
            scores[group] = current_loss - after_loss
            losses[group] = after_loss
        selected = min(scores, key=lambda group: (-scores[group], group))
        stages.append({"prefix": list(order), "selected": selected,
                       "mean_signed_ce_gain": scores[selected],
                       "candidate_gains": {str(group): value for group, value in scores.items()}})
        order.append(selected)
        current_mask[selected] = True
        current_loss = losses[selected]
    return tuple(order), {
        "method": "GLOBAL_GREEDY_STATIC_PREFIX", "fit_split": "policy_fit",
        "rows_used": len(records), "aggregation": "row-weighted training mean",
        "probability_floor_for_log": 1e-12,
        "test_labels_used": False, "stages": stages,
        "note": "One frozen global order; validation may choose its stopping prefix. Not a Matryoshka reproduction.",
    }


class EmpiricalLookaheadPolicy:
    """Small-K conditional empirical MDP with finite-depth or exhaustive search.

    Runtime arguments contain only H, candidates, remaining declared budget and a
    frozen cost function. Training labels/complete responses are stored parameters
    of this empirical model, never the current test row or test-specific oracle.
    """

    @classmethod
    def fit(cls, rows, head, schema, depth=2, max_groups=7,
            include_pairs=True, include_all=True, pairs=None, max_states=20000):
        if type(max_groups) is not int or max_groups < 1 or schema.num_groups > max_groups:
            raise ValueError("empirical lookahead is intentionally restricted to small K <= max_groups")
        if depth is not None and (type(depth) is not int or depth < 1):
            raise ValueError("depth must be positive integer or None for exhaustive empirical DP")
        if type(max_states) is not int or max_states < 1:
            raise ValueError("max_states must be positive")
        records = training_rows(rows, "policy_fit", schema)
        result = cls()
        result.schema, result.head = schema, head
        result.depth = depth
        result.include_pairs, result.include_all = include_pairs, include_all
        result.pairs = None if pairs is None else tuple(tuple(pair) for pair in pairs)
        # Validate the candidate pool now rather than at a late recursive state.
        candidate_actions(schema.empty_state(), schema, include_pairs, include_all, result.pairs)
        result._records = tuple(records)
        result.max_states = max_states
        result.report = {
            "method": "EMPIRICAL_MODEL_BASED_LOOKAHEAD", "fit_split": "policy_fit",
            "rows_used": len(records), "depth": "exact_empirical" if depth is None else depth,
            "max_groups": max_groups, "max_search_states": max_states,
            "unseen_history_rule": "global policy_fit response/label distribution, preserve observed H",
            "aggregation": "row-weighted empirical conditional distribution",
            "population_bayes_optimal": False, "reproduces_aco_or_brig": False,
            "limitations": ["sparse conditional support; no population risk guarantee",
                            "exact only for empirical model, budget, and fixed candidate actions",
                            "large-K requests rejected instead of silently approximated"],
        }
        return result

    def action_values(self, observed, actions, remaining_budget, declared_cost, cost_weight=0.0,
                      *, remaining_groups=None):
        observed = validate_observed(observed, self.schema)
        actions = tuple(validate_action(action, observed, self.schema) for action in actions)
        if (type(remaining_budget) not in (float, int) or math.isnan(remaining_budget)
                or remaining_budget < 0 or type(cost_weight) not in (float, int)
                or not math.isfinite(cost_weight) or cost_weight < 0):
            raise ValueError("budget must be nonnegative (positive infinity allowed); cost weight must be finite nonnegative")
        unqueried_groups = sum(not acquired for acquired in self.schema.group_mask(observed))
        if remaining_groups is None:
            remaining_groups = unqueried_groups
        if type(remaining_groups) is not int or not 0 <= remaining_groups <= unqueried_groups:
            raise ValueError("remaining_groups must be an integer within the unqueried-group count")
        # Caches are local to a decision call: no prior test examples update the
        # training distribution, budgets or parameters for subsequent patients.
        support_cache, terminal_cache, memo = {}, {}, {}

        def support(state):
            if state not in support_cache:
                if len(support_cache) >= self.max_states:
                    raise ValueError("empirical DP exceeded max_states including terminal histories")
                known = tuple((atom, value) for atom, value in enumerate(state) if value >= 0)
                matching = tuple(index for index, (z, _) in enumerate(self._records)
                                 if all(z[atom] == value for atom, value in known))
                support_cache[state] = matching or tuple(range(len(self._records)))
            return support_cache[state]

        def terminal(state):
            if state not in terminal_cache:
                prediction = self.head.predict(state)
                matches = support(state)
                terminal_cache[state] = sum(self._records[i][1] != prediction for i in matches) / len(matches)
            return terminal_cache[state]

        def action_cost(state, action):
            # Crucial: setup/shared-prefix charges depend on EACH simulated H,
            # not only on the initial patient's observed history.
            value = declared_cost(state, action)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("declared cost function returned invalid cost")
            if not action and value != 0:
                raise ValueError("this baseline requires STOP incremental cost zero")
            return float(value)

        def q_value(state, action, budget, depth_left, groups_left):
            if len(action) > groups_left:
                return math.inf
            cost = action_cost(state, action)
            if cost > budget + 1e-12:
                return math.inf
            if not action:
                return terminal(state)
            atoms = self.schema.expand(action)
            transitions = Counter(tuple(self._records[i][0][atom] for atom in atoms) for i in support(state))
            total = sum(transitions.values())
            expected = 0.0
            for values, count in transitions.items():
                next_state = list(state)
                for atom, value in zip(atoms, values):
                    next_state[atom] = value
                next_state = tuple(next_state)
                groups_after = groups_left - len(action)
                future = (terminal(next_state) if depth_left <= 1 or groups_after == 0 else
                          value_function(next_state, max(0.0, budget - cost), depth_left - 1, groups_after))
                expected += (count / total) * future
            return cost_weight * cost + expected

        def value_function(state, budget, depth_left, groups_left):
            key = state, budget, depth_left, groups_left
            if key not in memo:
                if len(memo) >= self.max_states:
                    raise ValueError("empirical DP exceeded max_states; reduce budget/depth instead of silently changing baseline")
                # Reserve before recursion so the bound includes active states.
                memo[key] = None
                options = candidate_actions(state, self.schema, self.include_pairs, self.include_all, self.pairs)
                options = tuple(action for action in options if len(action) <= groups_left)
                memo[key] = min(q_value(state, action, budget, depth_left, groups_left) for action in options)
            return memo[key]

        depth = self.depth if self.depth is not None else self.schema.num_groups
        return tuple(q_value(observed, action, float(remaining_budget), depth, remaining_groups) for action in actions)

    def choose(self, observed, actions, remaining_budget, declared_cost, cost_weight=0.0,
               *, remaining_groups=None):
        actions = tuple(actions)
        if not actions:
            raise ValueError("a nonempty candidate list including STOP is required")
        values = self.action_values(observed, actions, remaining_budget, declared_cost, cost_weight,
                                    remaining_groups=remaining_groups)
        index = min(range(len(actions)), key=lambda i: (values[i], bool(actions[i]),
                                                        declared_cost(observed, actions[i]), actions[i]))
        if not math.isfinite(values[index]):
            raise ValueError("no feasible action; include STOP")
        return actions[index]
