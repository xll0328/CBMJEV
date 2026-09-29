#!/usr/bin/env python3
"""Deterministic CPU sanity checks for the analytical extensions.

Run ``python3 tools/check_theory_extensions.py``. Only the standard library is
required. Finite probabilities and algebraic inequalities use exact Fractions;
entropy calculations use floating point. These checks are not mathematical
proofs, fitted-model error certificates, or empirical benchmark results.
"""

from fractions import Fraction as F
import itertools
import math
import random
import unittest


SEED = 20260929
STOP = -1
BITS = (0, 1)
ACTIONS = (0, 1, 2)
TOL = 1e-11


def normalize(values):
    total = sum(values)
    return tuple(F(value) / total for value in values)


def add_label_mass(table, key, label, mass):
    table.setdefault(key, [F(0), F(0)])[label] += mass


def bayes_risk(joint, observation):
    """Exact binary zero-one Bayes risk for a supplied observation map."""
    rows = {}
    for atom, mass in joint.items():
        add_label_mass(rows, observation(atom), atom[0], mass)
    return sum(min(row) for row in rows.values())


def noisy_addressing_joint(address_noise, payload_noise):
    """P(Y,U,V0,V1) with mutually independent fair S,X0,X1 and noises."""
    joint = {}
    for s, x0, x1, ns, n0, n1 in itertools.product(BITS, repeat=6):
        mass = F(1, 8)
        for noise, probability in ((ns, address_noise), (n0, payload_noise), (n1, payload_noise)):
            mass *= probability if noise else 1 - probability
        if mass:
            atom = ((x0, x1)[s], s ^ ns, x0 ^ n0, x1 ^ n1)
            joint[atom] = joint.get(atom, F(0)) + mass
    return joint


def legal_histories():
    """Every ordered, non-repeating binary-query history of length at most 3."""
    for length in range(4):
        for actions in itertools.permutations(ACTIONS, length):
            for responses in itertools.product(BITS, repeat=length):
                yield tuple(zip(actions, responses))


def random_joint(rng):
    atoms = tuple(itertools.product(BITS, repeat=4))
    return dict(zip(atoms, normalize([rng.randint(1, 19) for _ in atoms])))


def random_policy(rng, randomized):
    """A legal history-only policy; random choices have independent coins."""
    policy = {}
    for history in legal_histories():
        acquired = {action for action, _ in history}
        available = (STOP,) + tuple(action for action in ACTIONS if action not in acquired)
        if randomized:
            policy[history] = dict(zip(available, normalize([rng.randint(1, 7) for _ in available])))
        else:
            # Keep deterministic examples nontrivial without forcing their length.
            choice = rng.choice(available[1:] if not history else available)
            policy[history] = {choice: F(1)}
    return policy


def enumerate_tree(joint, policy):
    """Enumerate the exact reached-history and terminal-history label masses."""
    reached, terminal = {}, {}

    def visit(atom, mass, history):
        add_label_mass(reached, history, atom[0], mass)
        for action, probability in policy[history].items():
            branch_mass = mass * probability
            if not branch_mass:
                continue
            if action == STOP:
                add_label_mass(terminal, history, atom[0], branch_mass)
            else:
                if action not in ACTIONS or action in dict(history):
                    raise ValueError("A legal action queries one previously unacquired coordinate")
                visit(atom, branch_mass, history + ((action, atom[action + 1]),))

    for atom, mass in joint.items():
        if mass:
            visit(atom, mass, ())
    return reached, terminal


def evidence_label_mass(joint, history):
    """Condition only on acquired indexed values in the latent response vector."""
    labels = [F(0), F(0)]
    for atom, mass in joint.items():
        if all(atom[action + 1] == response for action, response in history):
            labels[atom[0]] += mass
    return labels


def label_entropy(label_masses):
    return -math.fsum(float(p) * math.log(float(p)) for p in normalize(label_masses) if p)


def query_information(joint, history, action):
    """I(Y; Z_action | the acquired values), by finite conditional enumeration."""
    before = evidence_label_mass(joint, history)
    mass = sum(before)
    after_entropy = 0.0
    for response in BITS:
        after = evidence_label_mass(joint, history + ((action, response),))
        if sum(after):
            after_entropy += float(sum(after) / mass) * label_entropy(after)
    return label_entropy(before) - after_entropy


class TheoryExtensionTests(unittest.TestCase):
    def test_noisy_addressing_all_fixed_pairs_adaptation_and_full_observation(self):
        grid = (F(0), F(1, 20), F(1, 4), F(9, 20), F(1, 2))
        for address_noise, payload_noise in itertools.product(grid, repeat=2):
            with self.subTest(address_noise=address_noise, payload_noise=payload_noise):
                joint = noisy_addressing_joint(address_noise, payload_noise)
                self.assertEqual(sum(joint.values()), 1)
                fixed_risk = F(1, 4) + payload_noise / 2
                for pair in itertools.combinations(ACTIONS, 2):
                    risk = bayes_risk(joint, lambda row: tuple((action, row[action + 1]) for action in pair))
                    self.assertEqual(risk, fixed_risk)
                adaptive_risk = bayes_risk(
                    joint, lambda row: ((0, row[1]), (1 + row[1], row[2 + row[1]]))
                )
                expected = (1 - address_noise) * payload_noise + address_noise / 2
                self.assertEqual(adaptive_risk, expected)
                self.assertEqual(bayes_risk(joint, lambda row: row[1:]), adaptive_risk)
                gap = (1 - 2 * address_noise) * (1 - 2 * payload_noise) / 4
                self.assertEqual(fixed_risk - adaptive_risk, gap)
                self.assertGreaterEqual(gap, 0)
                # All comparisons spend two equal-cost singleton acquisitions.
                self.assertEqual(fixed_risk + 2 * F(1, 7) - (adaptive_risk + 2 * F(1, 7)), gap)

    def test_payload_first_all_response_dependent_second_queries(self):
        grid = (F(0), F(1, 20), F(1, 4), F(9, 20), F(1, 2))
        for address_noise, payload_noise in itertools.product(grid, repeat=2):
            joint = noisy_addressing_joint(address_noise, payload_noise)
            for first in (1, 2):
                # Exhaust all four deterministic maps from the first bit to
                # either the address or the other payload. Independent
                # randomized maps are mixtures of these deterministic maps.
                for second_by_response in itertools.product((0, 3 - first), repeat=2):
                    def observation(row):
                        response = row[first + 1]
                        second = second_by_response[response]
                        return ((first, response), (second, row[second + 1]))

                    with self.subTest(e=address_noise, r=payload_noise, first=first, rule=second_by_response):
                        self.assertEqual(bayes_risk(joint, observation), F(1, 4) + payload_noise / 2)

    def test_legal_ordered_histories_have_acquired_value_posteriors(self):
        rng = random.Random(SEED)
        checked = 0
        for randomized in (False, True):
            for _ in range(25):
                joint = random_joint(rng)
                policy = random_policy(rng, randomized)
                reached, terminal = enumerate_tree(joint, policy)
                self.assertEqual(sum(sum(row) for row in terminal.values()), 1)
                for table in (reached, terminal):
                    for history, labels in table.items():
                        with self.subTest(randomized=randomized, history=history):
                            self.assertEqual(normalize(labels), normalize(evidence_label_mass(joint, history)))
                            checked += 1
        self.assertGreater(checked, 3000)

    def test_final_acquisition_mask_can_marginally_reveal_label(self):
        joint = {(y, y, z1, z2): F(1, 8) for y, z1, z2 in itertools.product(BITS, repeat=3)}
        policy = {history: {STOP: F(1)} for history in legal_histories()}
        policy[()] = {0: F(1)}
        policy[((0, 1),)] = {1: F(1)}
        _, terminal = enumerate_tree(joint, policy)
        masks = {}
        for history, labels in terminal.items():
            mask = frozenset(action for action, _ in history)
            for label, mass in enumerate(labels):
                add_label_mass(masks, mask, label, mass)
        self.assertEqual(normalize(evidence_label_mass(joint, ())), (F(1, 2), F(1, 2)))
        self.assertEqual(normalize(masks[frozenset((0,))]), (1, 0))
        self.assertEqual(normalize(masks[frozenset((0, 1))]), (0, 1))

    def test_order_dependent_response_counterexample_to_indexed_value_sufficiency(self):
        ordered, indexed = {}, {}
        for label, first in itertools.product(BITS, repeat=2):
            # The first response is Y for query 0 and 1-Y for query 1;
            # every second response is 0. There is no fixed latent Z vector.
            response = label if first == 0 else 1 - label
            history = ((first, response), (1 - first, 0))
            add_label_mass(ordered, history, label, F(1, 4))
            add_label_mass(indexed, tuple(sorted(history)), label, F(1, 4))
        self.assertEqual(normalize(ordered[((0, 0), (1, 0))]), (1, 0))
        self.assertEqual(normalize(ordered[((1, 0), (0, 0))]), (0, 1))
        self.assertEqual(normalize(indexed[((0, 0), (1, 0))]), (F(1, 2), F(1, 2)))

    def test_occupancy_mse_inequality_on_finite_random_instances(self):
        rng = random.Random(SEED + 1)
        for index in range(250):
            states, actions = rng.randint(1, 6), rng.randint(1, 5)
            mu = normalize([rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(states - 1)])
            p = (F(0), F(1, 7), F(1, 2), F(1))[index % 4]
            d_distribution = normalize([rng.randint(1, 9) if mass else 0 for mass in mu])
            d = tuple(p * mass for mass in d_distribution)
            coverage = max(occupied / reference for occupied, reference in zip(d, mu) if reference)
            behavior = [normalize([rng.randint(1, 9) for _ in range(actions)]) for _ in range(states)]
            beta = min(min(row) for row in behavior)
            errors = [[F(rng.randint(-9, 9), 5) for _ in range(actions)] for _ in range(states)]
            maxima = [max(map(abs, row)) for row in errors]
            mse = sum(mu[s] * behavior[s][a] * errors[s][a] ** 2 for s in range(states) for a in range(actions))
            occupied_error = sum(mass * error for mass, error in zip(d, maxima))
            with self.subTest(index=index, p=p):
                self.assertEqual(sum(d), p)
                self.assertLessEqual(p, 1)
                self.assertTrue(all(occupied <= coverage * reference for occupied, reference in zip(d, mu)))
                # Squaring compares the bound exactly, without sqrt rounding.
                self.assertLessEqual(occupied_error ** 2, p * coverage * mse / beta)
                self.assertLessEqual(float(occupied_error), math.sqrt(float(p * coverage * mse / beta)) + TOL)

    def test_occupancy_mse_bound_can_be_tight(self):
        mu, d = (F(1, 4), F(3, 4)), (F(1, 2), F(0))
        behavior = ((F(1, 4), F(3, 4)),) * 2
        errors = ((F(2), F(0)), (F(0), F(0)))
        p, coverage, beta = F(1, 2), F(2), F(1, 4)
        mse = sum(mu[s] * behavior[s][a] * errors[s][a] ** 2 for s in range(2) for a in range(2))
        occupied_error = sum(d[s] * max(map(abs, errors[s])) for s in range(2))
        self.assertEqual(occupied_error, 1)
        self.assertEqual(occupied_error ** 2, p * coverage * mse / beta)

    def test_missing_state_or_action_coverage_hides_error(self):
        # An occupied state outside reference support has error but zero MSE.
        mu, d, maxima = (F(1), F(0)), (F(0), F(1)), (F(0), F(1))
        self.assertEqual(sum(mu[s] * maxima[s] ** 2 for s in range(2)), 0)
        self.assertEqual(sum(d[s] * maxima[s] for s in range(2)), 1)
        # At a covered state, an action of behavior probability zero can also
        # hide the maximum error. The theorem requires strictly positive beta.
        behavior, errors = (F(1), F(0)), (F(0), F(1))
        self.assertEqual(sum(b * error ** 2 for b, error in zip(behavior, errors)), 0)
        self.assertEqual(max(errors), 1)

    def check_stop_certificates(self, immediate, continuation, upper, offsets, epsilon):
        self.assertEqual(continuation[0], 0)
        self.assertTrue(all(0 <= c <= u for c, u in zip(continuation, upper)))
        self.assertTrue(all(abs(offset) <= epsilon for offset in offsets))
        estimates = [score + offset for score, offset in zip(immediate, offsets)]
        q = [score - c for score, c in zip(immediate, continuation)]
        stop_certificate = all(estimates[a] - estimates[0] >= 2 * epsilon + upper[a] for a in range(1, len(q)))
        continue_certificate = any(estimates[0] - estimates[a] > 2 * epsilon for a in range(1, len(q)))
        if stop_certificate:
            self.assertEqual(q[0], min(q))
        if continue_certificate:
            self.assertGreater(q[0], min(q))
        return int(stop_certificate), int(continue_certificate)

    def test_stop_certificates_over_finite_grid_and_multiple_actions(self):
        hits = [0, 0]
        for epsilon, immediate, continuation, slack, stop_sign, query_sign in itertools.product(
            (F(0), F(1, 10), F(1, 4)),
            (F(i, 4) for i in range(9)),
            (F(0), F(1, 4), F(1, 2)),
            (F(0), F(1, 4)),
            (-1, 0, 1),
            (-1, 0, 1),
        ):
            if continuation > immediate:
                continue
            result = self.check_stop_certificates(
                (F(1), immediate), (F(0), continuation), (F(0), continuation + slack),
                (stop_sign * epsilon, query_sign * epsilon), epsilon,
            )
            hits = [old + new for old, new in zip(hits, result)]
        rng = random.Random(SEED + 2)
        for _ in range(250):
            count, epsilon = rng.randint(2, 6), F(rng.randint(0, 5), 10)
            continuation = [F(0)] + [F(rng.randint(0, 8), 10) for _ in range(count - 1)]
            upper = [c + F(rng.randint(0, 5), 10) for c in continuation]
            upper[0] = F(0)
            immediate = [F(rng.randint(0, 20), 10) + c for c in continuation]
            offsets = [epsilon * F(rng.randint(-5, 5), 5) for _ in range(count)]
            result = self.check_stop_certificates(immediate, continuation, upper, offsets, epsilon)
            hits = [old + new for old, new in zip(hits, result)]
        self.assertGreater(hits[0], 0)
        self.assertGreater(hits[1], 0)

    def test_stop_certificate_needs_continuation_upper_bound_and_all_actions(self):
        loss, immediate, continuation = F(1, 2), F(3, 5), F(3, 10)
        self.assertGreaterEqual(immediate - loss, 0)  # Exact myopic scores favor STOP.
        self.assertLess(immediate - continuation, loss)  # Continuing is better.
        self.assertLess(immediate - loss, continuation)  # Correct certificate fails.
        # Certifying one expensive acquisition is insufficient when another
        # acquisition beats STOP; the stopping condition quantifies over all.
        immediate = (F(1), F(2), F(0))
        self.assertGreaterEqual(immediate[1] - immediate[0], 0)
        self.assertGreater(immediate[0], min(immediate))

    def test_stop_certificate_boundary_and_strict_continuation_condition(self):
        epsilon, loss, upper = F(1, 10), F(1), F(3, 10)
        # Equality in the stopping certificate permits a genuine optimal tie.
        result = self.check_stop_certificates(
            (loss, loss + upper), (F(0), upper), (F(0), upper),
            (-epsilon, epsilon), epsilon,
        )
        self.assertEqual(result, (1, 0))
        # A reverse estimated gap of exactly 2*epsilon need not establish
        # strict suboptimality: both true Q values can coincide.
        estimates = (loss + epsilon, loss - epsilon)
        self.assertEqual(estimates[0] - estimates[1], 2 * epsilon)
        result = self.check_stop_certificates(
            (loss, loss), (F(0), F(0)), (F(0), F(0)), (epsilon, -epsilon), epsilon,
        )
        self.assertEqual(result, (0, 0))

    def test_pointwise_margin_regret_with_strictly_suboptimal_gap(self):
        values = (F(0), F(1, 4), F(1, 2), F(3, 4))
        for count in (1, 2, 3):
            for risks in itertools.product(values, repeat=count):
                optimal = min(risks)
                gamma = min((risk - optimal for risk in risks if risk > optimal), default=math.inf)
                for epsilon in (F(0), F(1, 8), F(1, 4), F(1, 2)):
                    for signs in itertools.product((-1, 0, 1), repeat=count):
                        estimates = [risk + sign * epsilon for risk, sign in zip(risks, signs)]
                        chosen = min(range(count), key=estimates.__getitem__)
                        regret = risks[chosen] - optimal
                        bound = 2 * epsilon if gamma <= 2 * epsilon else F(0)
                        self.assertLessEqual(regret, bound)

    def test_margin_ties_infinite_gap_and_sharp_threshold(self):
        risks, epsilon = (F(1, 2), F(0), F(0)), F(1, 4)
        optimal = min(risks)
        gamma = min(risk - optimal for risk in risks if risk > optimal)
        estimates = (F(1, 4),) * 3
        chosen = min(range(3), key=estimates.__getitem__)
        self.assertEqual(gamma, 2 * epsilon)
        self.assertEqual(risks[chosen] - optimal, 2 * epsilon)
        # Optimal ties must not force the strictly-suboptimal gap to zero.
        self.assertGreater(gamma, 0)
        all_optimal = (F(1, 3),) * 3
        gamma = min((risk - min(all_optimal) for risk in all_optimal if risk > min(all_optimal)), default=math.inf)
        self.assertEqual(gamma, math.inf)
        self.assertEqual(max(all_optimal) - min(all_optimal), 0)

    def test_adaptive_stopped_information_chain(self):
        rng = random.Random(SEED + 3)
        for randomized in (False, True):
            for index in range(20):
                joint, policy = random_joint(rng), random_policy(rng, randomized)
                reached, terminal = enumerate_tree(joint, policy)
                prior_entropy = label_entropy(evidence_label_mass(joint, ()))
                terminal_entropy = math.fsum(float(sum(row)) * label_entropy(row) for row in terminal.values())
                transcript_information = prior_entropy - terminal_entropy
                accumulated_information = math.fsum(
                    float(sum(labels) * probability) * query_information(joint, history, action)
                    for history, labels in reached.items()
                    for action, probability in policy[history].items()
                    if action != STOP and probability
                )
                with self.subTest(randomized=randomized, index=index):
                    self.assertEqual(sum(sum(row) for row in terminal.values()), 1)
                    self.assertAlmostEqual(transcript_information, accumulated_information, delta=TOL)
                    self.assertGreaterEqual(accumulated_information, -TOL)
                    self.assertLessEqual(accumulated_information, prior_entropy + TOL)

    def test_addressing_information_is_gained_only_after_branching(self):
        joint = noisy_addressing_joint(F(0), F(0))
        policy = {history: {STOP: F(1)} for history in legal_histories()}
        policy[()] = {0: F(1)}
        for address in BITS:
            policy[((0, address),)] = {1 + address: F(1)}
        reached, terminal = enumerate_tree(joint, policy)
        self.assertAlmostEqual(query_information(joint, (), 0), 0, delta=TOL)
        for address in BITS:
            self.assertAlmostEqual(query_information(joint, ((0, address),), 1 + address), math.log(2), delta=TOL)
        self.assertEqual({len(history) for history in terminal}, {2})
        self.assertAlmostEqual(sum(float(sum(row)) * label_entropy(row) for row in terminal.values()), 0, delta=TOL)
        information = math.fsum(
            float(sum(labels) * probability) * query_information(joint, history, action)
            for history, labels in reached.items()
            for action, probability in policy[history].items()
            if action != STOP
        )
        self.assertAlmostEqual(information, math.log(2), delta=TOL)


if __name__ == "__main__":
    print(
        f"CPU-only finite analytical-extension checks; seed={SEED}.\n"
        "Sanity checks only: not proofs, fitted-model certificates, or benchmark results.",
        flush=True,
    )
    unittest.main(verbosity=2)
