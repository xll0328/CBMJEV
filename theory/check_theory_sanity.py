#!/usr/bin/env python3
"""Finite CPU sanity checks for the manuscript's analytical identities.

Run with Python 3 and the standard library only. All random instances use fixed
seeds. Passing these checks is not a mathematical proof, a learned-controller
error certificate, or an empirical benchmark result.
"""

from dataclasses import dataclass, field
import itertools
import math
import random
import unittest


SEED = 20260928
TOL = 1e-11
STOP = "STOP"


def normalized(values):
    total = math.fsum(values)
    return [value / total for value in values]


def entropy(probabilities):
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0)


def kl_divergence(truth, prediction):
    return math.fsum(p * math.log(p / q) for p, q in zip(truth, prediction) if p > 0)


def average(weights, values):
    return math.fsum(w * v for w, v in zip(weights, values))


@dataclass
class Node:
    loss: float
    # Each acquisition maps to (cost, [(probability, next state), ...]).
    actions: dict = field(default_factory=dict)
    offsets: dict = field(default_factory=dict)


def evaluate_tree(node, local_checks):
    """Exact finite-tree DP; no Monte Carlo trajectory estimates are used."""
    q = {STOP: node.loss}
    immediate = {STOP: node.loss}
    children = {}
    for action, (cost, transitions) in node.actions.items():
        children[action] = [(p, evaluate_tree(child, local_checks)) for p, child in transitions]
        q[action] = cost + math.fsum(p * result["optimal"] for p, result in children[action])
        immediate[action] = cost + math.fsum(p * child.loss for p, child in transitions)
    estimates = {a: score + node.offsets.get(a, 0.0) for a, score in immediate.items()}
    chosen = min(estimates, key=estimates.get)  # Fixed insertion-order tie rule.
    optimal_action = min(q, key=q.get)
    optimal = q[optimal_action]
    continuation = {a: immediate[a] - q[a] for a in q}
    epsilon = max(abs(estimates[a] - immediate[a]) for a in q)
    oscillation = max(continuation.values()) - min(continuation.values())
    advantage = q[chosen] - optimal
    local_bound = 2 * epsilon + oscillation
    refined_bound = 2 * epsilon + continuation[optimal_action] - continuation[chosen]
    local_checks.append((continuation, advantage, refined_bound, local_bound))
    if chosen == STOP:
        return {
            "optimal": optimal, "policy": node.loss,
            "advantages": advantage, "bound": local_bound,
            "acquisition_only_advantages": 0.0,
        }
    cost = node.actions[chosen][0]
    future = children[chosen]
    return {
        "optimal": optimal,
        "policy": cost + math.fsum(p * result["policy"] for p, result in future),
        "advantages": advantage + math.fsum(p * result["advantages"] for p, result in future),
        "bound": local_bound + math.fsum(p * result["bound"] for p, result in future),
        "acquisition_only_advantages": advantage + math.fsum(
            p * result["acquisition_only_advantages"] for p, result in future
        ),
    }


def random_tree(rng, depth):
    node = Node(rng.random())
    if depth:
        for action in ("a", "b"):
            weights = normalized([0.1 + rng.random(), 0.1 + rng.random()])
            node.actions[action] = (
                0.2 * rng.random(),
                [(weight, random_tree(rng, depth - 1)) for weight in weights],
            )
    node.offsets = {action: rng.uniform(-0.25, 0.25) for action in (STOP, *node.actions)}
    return node


def joint_posteriors(joint, label_count):
    """Return true distributions and masses for a finite joint P(h,z,y)."""
    before, after = {}, {}
    for (h, z, y), mass in joint.items():
        before.setdefault(h, [0.0] * label_count)[y] += mass
        after.setdefault((h, z), [0.0] * label_count)[y] += mass
    before_mass = {h: math.fsum(row) for h, row in before.items()}
    after_mass = {hz: math.fsum(row) for hz, row in after.items()}
    return (
        {h: normalized(row) for h, row in before.items()},
        {hz: normalized(row) for hz, row in after.items()},
        before_mass, after_mass,
    )


def information_terms(joint, prediction_before, prediction_after, label_count):
    truth_before, truth_after, before_mass, after_mass = joint_posteriors(joint, label_count)
    ce_before = -math.fsum(
        mass * math.log(prediction_before[h][y])
        for (h, z, y), mass in joint.items() if mass > 0
    )
    ce_after = -math.fsum(
        mass * math.log(prediction_after[h, z][y])
        for (h, z, y), mass in joint.items() if mass > 0
    )
    information = (
        math.fsum(before_mass[h] * entropy(row) for h, row in truth_before.items())
        - math.fsum(after_mass[hz] * entropy(row) for hz, row in truth_after.items())
    )
    before_kl = math.fsum(
        before_mass[h] * kl_divergence(row, prediction_before[h]) for h, row in truth_before.items()
    )
    after_kl = math.fsum(
        after_mass[hz] * kl_divergence(row, prediction_after[hz]) for hz, row in truth_after.items()
    )
    return ce_before - ce_after, information, before_kl, after_kl


class TheorySanityTests(unittest.TestCase):
    def assert_close(self, actual, expected):
        self.assertAlmostEqual(actual, expected, delta=TOL)

    def test_conditional_ranking_identity_and_envelope(self):
        rng = random.Random(SEED)
        for _ in range(500):
            count, actions = rng.randint(1, 7), rng.randint(1, 6)
            weights = normalized([0.1 + rng.random() for _ in range(count)])
            risks = [[rng.uniform(-1, 2) for _ in range(actions)] for _ in range(count)]
            estimates = [[q + rng.uniform(-0.3, 0.3) for q in row] for row in risks]
            error = [max(abs(q - hat) for q, hat in zip(row, hats)) for row, hats in zip(risks, estimates)]
            selected = [min(range(actions), key=hats.__getitem__) for hats in estimates]
            static = min(average(weights, [row[a] for row in risks]) for a in range(actions))
            oracle = average(weights, [min(row) for row in risks])
            fitted = average(weights, [row[a] for row, a in zip(risks, selected)])
            gap = static - oracle
            self.assertGreaterEqual(gap, -TOL)
            self.assertGreaterEqual(fitted - oracle, -TOL)
            self.assertLessEqual(fitted - oracle, 2 * average(weights, error) + TOL)
            self.assertGreaterEqual(static - fitted, gap - 2 * average(weights, error) - TOL)
            self.assertLessEqual(static - fitted, gap + TOL)
            if actions == 2:
                differences = [row[0] - row[1] for row in risks]
                self.assert_close(gap, (average(weights, list(map(abs, differences))) - abs(average(weights, differences))) / 2)

    def test_ranking_ties_equality_and_sharp_constant(self):
        risks = [(0.0, 1.0), (0.0, 0.0)]
        self.assert_close(min(sum(row[a] for row in risks) / 2 for a in range(2)), 0)
        self.assert_close(sum(min(row) for row in risks) / 2, 0)
        error = 0.2
        q, estimated = [2 * error, 0.0], [error, error]
        chosen = min(range(2), key=estimated.__getitem__)
        self.assert_close(q[chosen] - min(q), 2 * error)

    def test_finite_tree_local_and_total_regret(self):
        rng = random.Random(SEED + 1)
        for _ in range(150):
            checks = []
            result = evaluate_tree(random_tree(rng, depth=3), checks)
            for continuation, advantage, refined, bound in checks:
                self.assert_close(continuation[STOP], 0)
                self.assertGreaterEqual(min(continuation.values()), -TOL)
                self.assertGreaterEqual(advantage, -TOL)
                self.assertLessEqual(advantage, refined + TOL)
                self.assertLessEqual(refined, bound + TOL)
            regret = result["policy"] - result["optimal"]
            self.assert_close(regret, result["advantages"])
            self.assertLessEqual(regret, result["bound"] + TOL)

    def test_terminal_stop_must_be_counted(self):
        root = Node(1.0, {"query": (0.0, [(1.0, Node(0.0))])}, {STOP: -1.0, "query": 1.0})
        result = evaluate_tree(root, [])
        self.assert_close(result["policy"] - result["optimal"], 1.0)
        self.assert_close(result["advantages"], 1.0)
        self.assert_close(result["acquisition_only_advantages"], 0.0)

    def test_exact_immediate_scores_can_still_be_myopic(self):
        # The conditional-risk process of noiseless XOR: .5 -> .5 -> 0.
        root = Node(0.5, {"first": (0.1, [(1.0, Node(0.5, {"second": (0.1, [(1.0, Node(0.0))])}))])})
        checks = []
        result = evaluate_tree(root, checks)
        self.assert_close(result["policy"], 0.5)
        self.assert_close(result["optimal"], 0.2)
        self.assert_close(result["advantages"], 0.3)
        self.assertLessEqual(0.3, result["bound"] + TOL)

    def test_noisy_parity_by_exact_enumeration(self):
        for rho in (0.0, 0.01, 0.1, 0.25, 0.49, 0.5):
            with self.subTest(rho=rho):
                joint = {}
                for c1, c2, e1, e2 in itertools.product((0, 1), repeat=4):
                    mass = 0.25 * (rho if e1 else 1 - rho) * (rho if e2 else 1 - rho)
                    key = (c1 ^ c2, c1 ^ e1, c2 ^ e2)
                    joint[key] = joint.get(key, 0.0) + mass
                d = 2 * rho * (1 - rho)
                for visible in ((), (1,), (2,), (1, 2)):
                    observed = {}
                    for row, mass in joint.items():
                        observed.setdefault(tuple(row[j] for j in visible), [0.0, 0.0])[row[0]] += mass
                    risk = math.fsum(min(row) for row in observed.values())
                    ce = math.fsum(math.fsum(row) * entropy(normalized(row)) for row in observed.values())
                    self.assert_close(risk, d if len(visible) == 2 else 0.5)
                    self.assert_close(ce, entropy([d, 1 - d]) if len(visible) == 2 else math.log(2))
                self.assert_close(0.5 - d, 0.5 * (1 - 2 * rho) ** 2)

    def test_information_kl_identity_with_misspecified_and_bayes_heads(self):
        rng = random.Random(SEED + 2)
        keys = list(itertools.product(range(3), range(3), range(4)))
        for _ in range(150):
            joint = dict(zip(keys, normalized([0.01 + rng.random() for _ in keys])))
            before = {h: normalized([0.01 + rng.random() for _ in range(4)]) for h in range(3)}
            after = {(h, z): normalized([0.01 + rng.random() for _ in range(4)]) for h in range(3) for z in range(3)}
            gain, information, kl_before, kl_after = information_terms(joint, before, after, 4)
            self.assert_close(gain, information + kl_before - kl_after)
            self.assertGreaterEqual(information, -TOL)
            truth_before, truth_after, _, _ = joint_posteriors(joint, 4)
            gain, information, kl_before, kl_after = information_terms(joint, truth_before, truth_after, 4)
            self.assert_close(gain, information)
            self.assert_close(kl_before, 0)
            self.assert_close(kl_after, 0)

    def test_negative_sample_targets_and_negative_expected_value(self):
        joint = {(0, z, y): 0.5 * (0.9 if z == y else 0.1) for z, y in itertools.product((0, 1), repeat=2)}
        before, after, _, _ = joint_posteriors(joint, 2)
        targets = [(mass, math.log(after[h, z][y] / before[h][y])) for (h, z, y), mass in joint.items()]
        self.assert_close(min(value for _, value in targets), math.log(0.2))
        mean_gain = math.fsum(mass * value for mass, value in targets)
        self.assert_close(mean_gain, math.log(2) - entropy([0.1, 0.9]))
        self.assertGreater(mean_gain, 0)
        self.assertGreater(math.fsum(mass * max(0, value) for mass, value in targets), mean_gain)
        independent = {(0, z, y): 0.25 for z, y in itertools.product((0, 1), repeat=2)}
        gain, information, kl_before, kl_after = information_terms(
            independent, {0: [0.5, 0.5]}, {(0, z): [0.9, 0.1] for z in (0, 1)}, 2
        )
        self.assert_close(information, 0)
        self.assert_close(gain, kl_before - kl_after)
        self.assertLess(gain, 0)


if __name__ == "__main__":
    print(
        f"CPU-only finite numerical checks; base seed={SEED}.\n"
        "Not mathematical proofs, error certificates, or empirical benchmark results.",
        flush=True,
    )
    unittest.main(verbosity=2)
