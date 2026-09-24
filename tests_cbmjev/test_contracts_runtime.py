import io
from pathlib import Path
import tempfile
import unittest

from cbmjev.contracts import (Concept, QueryGroup, Schema, DeclaredCost,
                             ModelInput, candidate_actions, validate_rows)
from cbmjev.runtime import ReplayEnvironment, LiveEnvironment, run_episode, load_payload


def fixture_schema():
    return Schema("fixture", 2,
                  (Concept("a", "A", ("no", "yes")), Concept("b", "B", ("no", "yes")),
                   Concept("c", "C", ("red", "green", "blue"))),
                  (QueryGroup("pair", (0, 1)), QueryGroup("colour", (2,))))


class Head:
    def probabilities(self, observed):
        return (0.8, 0.2) if observed[0] != 1 else (0.2, 0.8)


class Risk:
    objective = "risk"
    def predict(self, observed, actions):
        return tuple(0.6 if not a and observed[0] == -1 else
                     (0.1 if a == (0,) else 0.3) for a in actions)


class Responder:
    def __init__(self):
        self.seen = []
    def respond(self, payload, atoms):
        self.seen.append(payload)
        return tuple(1 for _ in atoms)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.schema = fixture_schema()

    def test_roundtrip(self):
        self.assertEqual(self.schema, Schema.from_dict(self.schema.to_dict()))
        self.assertEqual(self.schema.num_categories, (4, 4, 5))

    def test_partition_and_partial_group_rejected(self):
        with self.assertRaises(ValueError):
            Schema("x", 2, self.schema.concepts, (QueryGroup("x", (0, 1)),))
        with self.assertRaises(ValueError):
            self.schema.validate_state((0, -1, -1))

    def test_groups_not_atoms_are_charged(self):
        result = run_episode(ReplayEnvironment((1, 0, 2), self.schema), self.schema,
                             Head(), method="fixed", max_groups=1)
        self.assertEqual(result["queried_groups"], [0])
        self.assertEqual(result["queried_atoms"], [0, 1])
        self.assertEqual(result["declared_cost"], 1)
        self.assertEqual(result["final_state"], [1, 0, -1])

    def test_stop_zero_calls(self):
        result = run_episode(ReplayEnvironment((1, 0, 2), self.schema), self.schema,
                             Head(), method="stop")
        self.assertEqual(result["calls"], 0)
        self.assertNotIn("total_wall_ms", result)

    def test_empty_budget(self):
        result = run_episode(ReplayEnvironment((1, 0, 2), self.schema), self.schema,
                             Head(), method="fixed", max_cost=0)
        self.assertEqual(result["calls"], 0)

    def test_all_is_one_full_batch_not_budgeted_largest(self):
        for options in ({"max_groups": 1}, {"max_cost": 0}, {"include_all": False}):
            with self.assertRaises(ValueError):
                run_episode(ReplayEnvironment((1, 0, 2), self.schema), self.schema,
                            Head(), method="all", **options)
        trace = run_episode(ReplayEnvironment((1, 0, 2), self.schema), self.schema, Head(), method="all")
        self.assertEqual(trace["calls"], 1)
        self.assertEqual(trace["queried_groups"], [0, 1])

    def test_live_input_mutation_refuses_frozen_cache_identity(self):
        from cbmjev.contracts import stable_hash
        env = LiveEnvironment({"modality": "text", "text": "changed"}, self.schema, Responder(),
                              expected_digest=stable_hash({"text": "original", "images": []}))
        with self.assertRaises(ValueError):
            run_episode(env, self.schema, Head(), method="all")

    def test_live_answer_drift_is_not_silently_dropped(self):
        env = LiveEnvironment({"modality": "text", "text": "same text"}, self.schema, Responder(),
                              expected_responses=(0, 0, 0))
        with self.assertRaises(ValueError):
            run_episode(env, self.schema, Head(), method="all")

    def test_shared_setup_only_paid_once(self):
        cost = DeclaredCost(setup=3, call=2, per_group=1)
        self.assertEqual(cost((-1, -1, -1), (0,)), 6)
        self.assertEqual(cost((0, 1, -1), (1,)), 3)
        self.assertEqual(cost((0, 1, -1), ()), 0)

    def test_hidden_values_do_not_change_first_action(self):
        traces = [run_episode(ReplayEnvironment(z, self.schema), self.schema, Head(),
                              method="risk", controller=Risk())
                  for z in ((1, 0, 2), (0, 1, 0))]
        self.assertEqual(traces[0]["steps"][0]["action"], traces[1]["steps"][0]["action"])

    def test_candidates_depend_on_purchased_groups_only(self):
        self.assertEqual(candidate_actions((0, 1, -1), self.schema),
                         candidate_actions((1, 0, -1), self.schema))
        self.assertEqual(candidate_actions((-1, -1, -1), self.schema),
                         ((), (0,), (1,), (0, 1)))

    def test_modelinput_metadata_rejected(self):
        with self.assertRaises(TypeError):
            ModelInput(text="hello", sample_id="label")
        with self.assertRaises(ValueError):
            ModelInput(text="hello", images=(b"abc",))

    def test_live_only_content_enters_responder(self):
        responder = Responder()
        env = LiveEnvironment({"modality": "text", "text": "food good", "image_paths": []},
                              self.schema, responder)
        trace = run_episode(env, self.schema, Head(), method="fixed", max_groups=1)
        self.assertEqual(responder.seen, [ModelInput(text="food good")])
        self.assertIn("total_wall_ms", trace)
        self.assertEqual(trace["calls"], 1)

    def test_same_pixels_filename_does_not_enter_model(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            picture = Image.new("RGB", (3, 3), "white")
            picture.save(root / "class_A.png")
            picture.save(root / "class_B.png")
            a = load_payload({"modality": "image", "image_paths": ["class_A.png"]}, root)
            b = load_payload({"modality": "image", "image_paths": ["class_B.png"]}, root)
            self.assertEqual(a, b)
            with self.assertRaises(ValueError):
                load_payload({"modality": "image", "image_paths": ["../outside.png"]}, root)

    def test_nonfinite_cost_rejected(self):
        with self.assertRaises(ValueError):
            DeclaredCost(per_group=float("nan"))

    def test_group_leakage_rejected(self):
        rows = [{"sample_id": str(i), "group_id": "shared", "split": role,
                 "z": [0, 0, 0], "y": 0} for i, role in enumerate(("head_fit", "test"))]
        with self.assertRaises(ValueError):
            validate_rows(rows, self.schema)


if __name__ == "__main__":
    unittest.main()
