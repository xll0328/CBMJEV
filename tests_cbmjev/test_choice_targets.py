import dataclasses
import math
import unittest

from cbmjev.choice_targets import singleton_choice_example
from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.decision_sets import DecisionInputs, DecisionMetadata, DecisionSet, RiskSupervision


class ChoiceTargetTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("synthetic", 2,
            (Concept("a", "a", ("no", "yes")), Concept("b", "b", ("no", "yes"))),
            (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        self.decision = DecisionSet(DecisionInputs((-1, -1), ((), (0,), (1,), (0, 1))),
            RiskSupervision((1., 0., 0., 0.)),
            DecisionMetadata("source", "digest", 0, 0, "sample", "group"))

    def make(self, decision=None, **options):
        kwargs = dict(remaining_groups=2, cost_weight=.2, temperature=.5)
        kwargs.update(options)
        return singleton_choice_example(decision or self.decision, self.schema, **kwargs)

    def test_formula_filter_and_no_privileged_forward_fields(self):
        example = self.make()
        self.assertEqual(example.model_inputs.actions, ((), (0,), (1,)))
        expected = [math.exp(-1/.5), math.exp(-.2/.5), math.exp(-.2/.5)]
        for actual, raw in zip(example.supervision.teacher_probabilities, expected):
            self.assertAlmostEqual(actual, raw/sum(expected))
        self.assertEqual(set(dataclasses.asdict(example.model_inputs)),
            {"observed", "actions", "remaining_groups", "incremental_costs", "cost_weight"})
        changed = dataclasses.replace(self.decision, supervision=RiskSupervision((0., 1., 0., 0.)))
        self.assertEqual(self.make(changed).model_inputs, example.model_inputs)
        self.assertNotEqual(self.make(changed).derived_id, example.derived_id)

    def test_budget_endpoints_ties_and_occurrence_identity(self):
        zero = self.make(remaining_groups=0)
        self.assertEqual(zero.model_inputs.actions, ((),))
        self.assertEqual(zero.supervision.teacher_probabilities, (1.,))
        tied = dataclasses.replace(self.decision, supervision=RiskSupervision((0., 0., 0., 0.)))
        self.assertEqual(self.make(tied, cost_weight=0).supervision.teacher_probabilities, (1/3,)*3)
        repeated = dataclasses.replace(self.decision,
            metadata=dataclasses.replace(self.decision.metadata, occurrence=1))
        self.assertNotEqual(self.make(repeated).derived_id, self.make().derived_id)

    def test_bad_target_and_partial_set_rejected_even_zero_budget(self):
        bad = dataclasses.replace(self.decision, supervision=RiskSupervision((.5, 0., 0., 0.)))
        with self.assertRaisesRegex(ValueError, "Boolean"):
            self.make(bad)
        partial = dataclasses.replace(self.decision,
            model_inputs=DecisionInputs((-1, -1), ((), (0,))), supervision=RiskSupervision((1., 0.)))
        with self.assertRaisesRegex(ValueError, "every required"):
            self.make(partial, remaining_groups=0)

    def test_public_costs_temperature_and_numeric_guards(self):
        result = self.make(group_costs=(0, 2), temperature=1e-300)
        self.assertEqual(result.supervision.teacher_probabilities, (0., 1., 0.))
        for options in (dict(cost_weight=True), dict(temperature=0), dict(remaining_groups=True),
                        dict(group_costs=[1]), dict(group_costs=[1, float("nan")]),
                        dict(cost_weight=1e308, group_costs=(1e308, 1))):
            with self.assertRaises(ValueError):
                self.make(**options)

    def test_t8_soft_teacher_can_invert_expected_action_risk(self):
        schema = Schema("t8", 2,
            tuple(Concept(str(g), str(g), ("0", "1")) for g in range(28)),
            tuple(QueryGroup(str(g), (g,)) for g in range(28)))
        actions = ((),) + tuple((g,) for g in range(28))
        # STOP, A, B, C, then 25 dummy singleton queries.
        errors_i = (1., 0., 1., 0.) + (1.,) * 25
        errors_ii = (1., 1., 0., 1.) + (1.,) * 25
        decisions = [DecisionSet(DecisionInputs(schema.empty_state(), actions),
            RiskSupervision(errors), DecisionMetadata("source", "digest", index, 0,
                                                     str(index), str(index)))
            for index, errors in enumerate((errors_i, errors_ii))]
        first, second = [singleton_choice_example(decision, schema,
            remaining_groups=28, cost_weight=.03, temperature=.5)
            for decision in decisions]
        p = .52
        mean_teacher = [p * a + (1-p) * b for a, b in zip(
            first.supervision.teacher_probabilities,
            second.supervision.teacher_probabilities)]
        self.assertEqual(max(range(29), key=mean_teacher.__getitem__), 2)
        self.assertAlmostEqual(mean_teacher[1], .1053733622205853)
        self.assertAlmostEqual(mean_teacher[2], .1124750729175609)
        self.assertLess(1-p+.03, p+.03)  # A has lower expected error + cost.

    def test_verified_oof_package_to_complete_question_training(self):
        import torch
        from cbmjev.decision_sets import iter_decision_sets
        from cbmjev.crossfit_training import construct_action_targets
        from cbmjev.nano_choice import NanoChoiceHead, choice_soft_target_loss
        from tests_cbmjev import test_crossfit_training as fixtures
        fixtures.NestedOOFTrainingPrimitiveTests.setUpClass()
        fixture = fixtures.NestedOOFTrainingPrimitiveTests
        config = {**fixtures.CONFIG, "actions_per_state": 64}
        package = construct_action_targets(fixture.target_rows, fixture.head, fixture.schema,
            config, head_report=fixture.head_report,
            response_artifact_by_group=fixture.target_assignments,
            provenance_records=[fixture.outer_record])
        original_hash = package["package_sha256"]
        decisions = iter_decision_sets(package, fixture.schema, config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(19)
            width = fixture.schema.num_atoms + fixture.schema.num_groups + 2
            model = NanoChoiceHead(width)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            count = 0
            for decision in decisions:
                example = singleton_choice_example(decision, fixture.schema,
                    remaining_groups=fixture.schema.num_groups, cost_weight=.1, temperature=.5)
                visible = example.model_inputs
                # Synthetic structured encoding uses ONLY the model-facing object.
                features = [[*visible.observed,
                    *(float(g in action) for g in range(fixture.schema.num_groups)),
                    visible.remaining_groups, visible.cost_weight] for action in visible.actions]
                x = torch.tensor([features], dtype=torch.float32)
                valid = torch.ones(x.shape[:2], dtype=torch.bool)
                teacher = torch.tensor([example.supervision.teacher_probabilities])
                loss = choice_soft_target_loss(model(x, valid), teacher, valid)
                optimizer.zero_grad()
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                optimizer.step()
                count += 1
                if count == 3:
                    break
            self.assertEqual(count, 3)
        self.assertEqual(original_hash, package["package_sha256"])


if __name__ == "__main__":
    unittest.main()
