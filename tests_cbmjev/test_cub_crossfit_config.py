"""Configuration wiring only; not visual fitting or quality evidence."""
import inspect
import json
from pathlib import Path
import unittest

from cbmjev.config import resolve_config
from cbmjev.pipeline import train_responder


class CubCrossfitConfigTests(unittest.TestCase):
    def test_explicit_visual_fit_settings_and_non_smoke_head(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        responder = json.loads((root / "cub_crossfit_responder_balanced_seed60.json").read_text())
        bound = inspect.signature(train_responder).bind("prepared", "out", **responder)
        bound.apply_defaults()
        self.assertEqual(bound.arguments["concept_class_weighting"], "inverse_frequency")
        self.assertEqual(bound.arguments["input_workers"], 2)
        self.assertEqual(bound.arguments["input_prefetch_batches"], 2)
        config = resolve_config(json.loads((root / "cub_crossfit_risk_seed60.json").read_text()))
        self.assertEqual(config["seed"], responder["seed"])
        self.assertEqual(config["device"], responder["device"])
        self.assertEqual(config["learning"]["head_epochs"], 50)
        self.assertEqual(config["learning"]["policy_epochs"], 30)
        self.assertEqual(config["learning"]["objective"], "risk")


if __name__ == "__main__":
    unittest.main()
