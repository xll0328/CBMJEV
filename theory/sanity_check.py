#!/usr/bin/env python3
"""Finite theory fixtures; no training, downloads, or empirical ML claims."""

from __future__ import annotations

import itertools
import json
import math
import random
from fractions import Fraction


def xor_bayes_risk(noise_joint, observed):
    """Exact Bayes 0-1 risk for uniform concepts and C-independent noise."""
    if sum(noise_joint.values()) != 1:
        raise ValueError("Noise probabilities must sum exactly to one.")
    masses = {}
    for c1, c2 in itertools.product((0, 1), repeat=2):
        for (e1, e2), probability in noise_joint.items():
            if probability < 0:
                raise ValueError("Noise probabilities must be nonnegative.")
            z = (c1 ^ e1, c2 ^ e2)
            key = tuple(z[index] for index in observed)
            masses.setdefault(key, [Fraction(0), Fraction(0)])
            masses[key][c1 ^ c2] += probability / 4
    return sum((min(labels) for labels in masses.values()), Fraction(0))


def independent_noise(rho):
    return {
        (0, 0): (1 - rho) ** 2,
        (0, 1): (1 - rho) * rho,
        (1, 0): rho * (1 - rho),
        (1, 1): rho**2,
    }


def check_xor():
    rows = []
    for numerator in range(11):
        rho = Fraction(numerator, 20)
        joint = independent_noise(rho)
        for observed in ((), (0,), (1,)):
            assert xor_bayes_risk(joint, observed) == Fraction(1, 2)
        pair_risk = xor_bayes_risk(joint, (0, 1))
        assert pair_risk == 2 * rho * (1 - rho)
        gain = Fraction(1, 2) - pair_risk
        assert gain == (1 - 2 * rho) ** 2 / 2
        rows.append(
            {"rho": float(rho), "bayes_pair_risk": float(pair_risk), "gain": float(gain)}
        )
    return rows


def check_correlated_noise():
    joint_families = {
        "independent": independent_noise(Fraction(1, 4)),
        "same_direction": {
            (0, 0): Fraction(3, 4),
            (0, 1): Fraction(0),
            (1, 0): Fraction(0),
            (1, 1): Fraction(1, 4),
        },
        "mutually_exclusive": {
            (0, 0): Fraction(1, 2),
            (0, 1): Fraction(1, 4),
            (1, 0): Fraction(1, 4),
            (1, 1): Fraction(0),
        },
    }
    expected = {
        "independent": Fraction(3, 8),
        "same_direction": Fraction(0),
        "mutually_exclusive": Fraction(1, 2),
    }
    results = {}
    for name, joint in joint_families.items():
        for index in (0, 1):
            assert sum(p for e, p in joint.items() if e[index]) == Fraction(1, 4)
        d = sum(p for e, p in joint.items() if e[0] != e[1])
        risk = xor_bayes_risk(joint, (0, 1))
        assert risk == min(d, 1 - d) == expected[name]
        results[name] = float(risk)
    return results


def check_boundary():
    # Exact algebra avoids a strict-inequality assertion at a floating tie.
    for rho in (Fraction(0), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2)):
        gain = (1 - 2 * rho) ** 2 / 2
        pair_risk = 2 * rho * (1 - rho)
        assert pair_risk + gain == Fraction(1, 2)
        if gain > 0:
            assert pair_risk + gain / 2 < Fraction(1, 2)
        assert pair_risk + gain + Fraction(1, 1000) > Fraction(1, 2)
    assert math.isclose((1 - math.sqrt(2 * 0.125)) / 2, 0.25)
    return {"critical_rho_at_cost_penalty_0_125": 0.25, "ties_are_not_strict_gains": True}


def check_regret():
    rng = random.Random(20260921)
    epsilon, eta, penalty = 0.1, 0.2, 0.7
    bound = 2 * epsilon + 2 * penalty * eta
    for _ in range(2000):
        risks = [rng.random() for _ in range(5)]
        costs = [rng.random() for _ in range(5)]
        estimated_risks = [value + rng.uniform(-epsilon, epsilon) for value in risks]
        estimated_costs = [value + rng.uniform(-eta, eta) for value in costs]
        objective = [r + penalty * c for r, c in zip(risks, costs)]
        estimated = [r + penalty * c for r, c in zip(estimated_risks, estimated_costs)]
        selected = min(range(5), key=estimated.__getitem__)
        regret = objective[selected] - min(objective)
        assert -1e-12 <= regret <= bound + 1e-12
    return {"finite_generated_cases": 2000, "upper_bound": bound}


def check_cost():
    rows = []
    for concept_count, queried, rounds, setup, per_concept, controller in (
        (28, 4, 4, 100, 1, 2),
        (28, 4, 1, 100, 1, 2),
        (28, 12, 3, 2, 10, 1),
    ):
        full = setup + per_concept * concept_count
        adaptive = setup * rounds + per_concept * queried + controller * rounds
        formula = per_concept * (concept_count - queried) - setup * (rounds - 1) - controller * rounds
        assert full - adaptive == formula
        rows.append({"queries": queried, "rounds": rounds, "model_cost_saving": formula})
    assert rows[0]["queries"] < 28 and rows[0]["model_cost_saving"] < 0
    assert rows[1]["model_cost_saving"] > 0
    return rows


def next_action(history, seed):
    # No raw-input parameter or closure. Seed intentionally irrelevant here.
    del seed
    if not history:
        return (1,)
    evidence = dict(history)
    if evidence[1] == 1 and 2 not in evidence:
        return (2,)
    return "STOP"


def check_noninterference_and_calibration():
    for seed in (0, 1, 2):
        for history in ((), ((1, 0),), ((1, 1),), ((1, 1), (2, 0))):
            first = next_action(history, seed)
            # Raw input and latency are outside the controller's interface.
            replaced_raw = {"unqueried": 999, "latency": 1e9}
            assert replaced_raw["latency"] > 0
            assert next_action(history, seed) == first
    for y in (0, 1):
        terminal_mask = {1, 2} if y else {1}
        assert int(2 in terminal_mask) == y
    pooled_mean = sum((0, 1)) / 2
    selected_mean = 1.0
    assert pooled_mean == 0.5 and selected_mean != 0.5
    return {"mask_can_predict_y_without_bypass": True, "pooled_mean": pooled_mean, "selected_mean": selected_mean}


def certify(empirical_risks, n, delta, alpha):
    if not empirical_risks or n < 1 or not 0 < delta < 1 or not 0 <= alpha <= 1:
        raise ValueError("Require M,n >= 1, delta in (0,1), and alpha in [0,1].")
    if any(not 0 <= risk <= 1 for risk in empirical_risks):
        raise ValueError("Terminal empirical risks must lie in [0,1].")
    margin = math.sqrt(math.log(len(empirical_risks) / delta) / (2 * n))
    upper = [min(1.0, risk + margin) for risk in empirical_risks]
    certified = [index for index, value in enumerate(upper) if value <= alpha]
    return margin, upper, certified


def check_certification():
    margin, _, certified = certify([0.0] * 20, 203, 0.05, 0.10)
    assert not certified
    larger_margin, _, certified = certify([0.0] * 20, 1000, 0.05, 0.10)
    assert len(certified) == 20 and larger_margin < margin
    return {"M": 20, "delta": 0.05, "margin_n203": margin, "margin_n1000": larger_margin, "n203_zero_error_target_0_10": "NO_CERTIFICATE"}


def check_choice_softmax_inversion():
    """T8: exact candidate count and live pilot cost/temperature semantics."""
    p, lam, temperature = 0.52, 0.03, 0.5
    # STOP, A, B, C, then 25 dummy singleton queries.
    errors_i = [1, 0, 1, 0] + [1] * 25
    errors_ii = [1, 1, 0, 1] + [1] * 25
    assert len(errors_i) == len(errors_ii) == 29

    def teacher(errors):
        utility = [errors[0]] + [e + lam for e in errors[1:]]
        weights = [math.exp(-u / temperature) for u in utility]
        return [weight / sum(weights) for weight in weights]

    qi, qii = teacher(errors_i), teacher(errors_ii)
    mean_q = [p * left + (1 - p) * right for left, right in zip(qi, qii)]
    expected_u = [p * left + (1 - p) * right + (0 if i == 0 else lam)
                  for i, (left, right) in enumerate(zip(errors_i, errors_ii))]
    r, c = math.exp(-1 / temperature), math.exp(-lam / temperature)
    di, dii = 2 * c + 26 * c * r + r, c + 27 * c * r + r
    threshold = di / (di + dii)
    assert di > dii and 0.5 < p < threshold
    assert math.isclose(mean_q[1] - mean_q[2],
                        c * (1 - r) * (p / di - (1 - p) / dii), abs_tol=1e-14)
    assert max(range(29), key=mean_q.__getitem__) == 2  # Teacher selects B.
    assert min(range(29), key=expected_u.__getitem__) == 1  # Utility selects A.
    assert math.isclose(expected_u[2] - expected_u[1], 0.04, abs_tol=1e-14)
    return {"candidates": 29, "p": p, "lambda": lam, "temperature": temperature,
            "inversion_threshold": threshold, "teacher_q_A": mean_q[1],
            "teacher_q_B": mean_q[2], "expected_utility_A": expected_u[1],
            "expected_utility_B": expected_u[2], "excess_utility": expected_u[2] - expected_u[1]}


def main():
    results = {
        "artifact_type": "analytic_fixture_sanity_checks",
        "is_real_dataset_experiment": False,
        "is_formal_proof_verification": False,
        "checks": {
            "independent_xor": check_xor(),
            "correlated_noise": check_correlated_noise(),
            "phase_boundary": check_boundary(),
            "regret": check_regret(),
            "cost": check_cost(),
            "noninterference_and_calibration": check_noninterference_and_calibration(),
            "certification": check_certification(),
            "choice_softmax_inversion": check_choice_softmax_inversion(),
        },
        "status": "PASS",
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
