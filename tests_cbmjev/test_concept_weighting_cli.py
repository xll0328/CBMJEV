import unittest
from unittest.mock import patch

from cbmjev import cli, pipeline


class ConceptWeightingCLITests(unittest.TestCase):
    def test_public_and_crossfit_dispatch_forward_weighting(self):
        for command, extra, target in (
            ("train-responder", [], "train_responder"),
            ("train-crossfit-responder", ["--planned", "plan", "--final"], "train_crossfit_responder"),
        ):
            args = cli.parser().parse_args([command, "--prepared", "prepared", "--out", "new",
                "--kind", "resnet18", "--concept-class-weighting", "inverse_frequency"] + extra)
            with patch.object(pipeline, target, return_value={}) as fit:
                cli.dispatch(args)
            self.assertEqual(fit.call_args.kwargs["concept_class_weighting"], "inverse_frequency")

    def test_unsupported_backends_fail_before_io(self):
        for kind in ("hashing_text", "nano_semantic"):
            with self.assertRaisesRegex(ValueError, "requires resnet18 or hf_text"):
                pipeline._train_selected_responder("missing", "new", None, [], kind=kind,
                    concept_class_weighting="inverse_frequency")


if __name__ == "__main__":
    unittest.main()
