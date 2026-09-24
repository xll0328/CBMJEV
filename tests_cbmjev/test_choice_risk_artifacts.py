from pathlib import Path
import tempfile
import unittest

from cbmjev.choice_artifacts import train_choice_pair
from cbmjev.io import read_json
from scripts.train_choice_risk_pair import MATCHED_KEYS, load_choice_risk_pair, train_choice_risk_pair
from tests_cbmjev.test_choice_artifacts import ChoiceArtifactTests


class ChoiceRiskArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ChoiceArtifactTests.setUpClass()
        cls.fixtures = ChoiceArtifactTests()

    def test_matched_artifact_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixtures.fixture(root)
            soft, risk = Path(root) / "soft", Path(root) / "risk"
            train_choice_pair(prepared, targets, config, soft, epochs=1,
                max_questions=8, batch_size=2, budget_mode="uniform_remaining")
            receipt = train_choice_risk_pair(prepared, targets, config, soft, risk)
            self.assertEqual(receipt["status"], "COMPLETE")
            scalar, attention, settings, report = load_choice_risk_pair(risk,
                prepared=prepared, targets=targets, config=config, soft_choice=soft)
            self.assertEqual(settings["objective"], "question-mean-active-utility-MSE")
            self.assertEqual(report["selected_occurrences"], 8)
            self.assertEqual(report["matched_keys"], list(MATCHED_KEYS))
            soft_fit = read_json(soft / "report.json")["fit"]
            for key in MATCHED_KEYS:
                self.assertEqual(report["fit"][key], soft_fit[key])
            self.assertFalse(any(p.requires_grad for p in scalar.parameters()))
            self.assertFalse(any(p.requires_grad for p in attention.parameters()))
            second = Path(root) / "risk_second"
            other = train_choice_risk_pair(prepared, targets, config, soft, second)
            self.assertEqual(read_json(risk / "report.json")["fit"],
                             read_json(second / "report.json")["fit"])
            self.assertEqual(receipt["bindings"], other["bindings"])
            with (risk / "scalar.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_choice_risk_pair(risk, prepared=prepared, targets=targets,
                    config=config, soft_choice=soft)


if __name__ == "__main__":
    unittest.main()
