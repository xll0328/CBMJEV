"""Real artifact roundtrip on synthetic software fixtures, not paper evidence."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest

from cbmjev import pipeline
from cbmjev.brig_artifacts import load_brig_controller
from cbmjev.io import read_json, read_jsonl


class BRiGArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="cbmjev-brig-test-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        pipeline.smoke(cls.root / "smoke", seed=17)
        cls.models = cls.root / "smoke/models"
        cls.cache = cls.root / "smoke/cache"
        cls.brig = cls.root / "brig"
        pipeline.train_brig_controller(cls.cache, cls.models, cls.brig,
                                       config=dict(hidden=8, max_budget=2, epochs=1))

    def test_roundtrip_and_budget(self):
        policy, report = load_brig_controller(self.brig, self.models, self.cache)
        self.assertTrue(report["head_exclusion_provenance_checked"])
        self.assertEqual(set(policy.models), {1, 2})
        for budget in (0, 1, 2):
            out = self.root / ("eval" + str(budget))
            pipeline.evaluate_models(self.models, self.cache, out, methods=["brig"],
                                     brig_dir=self.brig, max_groups=budget)
            for trace in read_jsonl(out / "traces.jsonl"):
                self.assertEqual(len(trace["queried_groups"]), budget)
                self.assertEqual(trace["calls"], budget)
                self.assertTrue(all(len(s["action"]) <= 1 for s in trace["steps"]))
            self.assertIsNotNone(read_json(out / "metrics.json")["brig_identity"])

    def test_formal_paths_rejected(self):
        for kwargs in (dict(split="test", evaluate_test=True), dict(certification_run=True)):
            with self.assertRaises(ValueError):
                pipeline.evaluate_models(self.models, self.cache, self.root / "reject",
                                         methods=["brig"], brig_dir=self.brig, **kwargs)
        with self.assertRaises(ValueError):
            pipeline.freeze_family(self.models, self.cache, self.root / "freeze", methods=["brig"])
        with self.assertRaises(ValueError):
            pipeline.evaluate_models(self.models, self.cache, self.root / "untrained",
                                     methods=["brig"], brig_dir=self.brig, max_groups=3)

    def test_receipt_and_parent_tampering(self):
        tampered = self.root / "tampered"
        shutil.copytree(self.brig, tampered)
        report = read_json(tampered / "training.json")
        report["head_component_sha256"] = "wrong"
        (tampered / "training.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "changed"):
            load_brig_controller(tampered, self.models, self.cache)
        parent = self.root / "changed-parent"
        shutil.copytree(self.models, parent)
        config = read_json(parent / "config.json")
        config["seed"] += 1
        (parent / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "binding"):
            load_brig_controller(self.brig, parent, self.cache)


if __name__ == "__main__":
    unittest.main()
