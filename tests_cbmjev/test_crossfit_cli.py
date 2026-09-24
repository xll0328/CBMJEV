"""Exercise public CLI boundaries with actual tiny concept-model training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cbmjev.cli import main, parser
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.io import read_json
from tests_cbmjev.test_crossfit import fixture, write_json


class CrossfitCLITests(unittest.TestCase):
    def test_evaluation_dispatch_is_validation_only(self):
        args = ["evaluate-crossfit", "--prepared", "p", "--planned", "f", "--merged", "m",
                "--responder", "r", "--cache", "c", "--out", "o"]
        with patch("cbmjev.crossfit_evaluation.evaluate_crossfit_validation", return_value={}) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            evaluate.assert_called_once_with("p", "f", "m", "r", "c", "o", methods=None,
                                              cost_weight=None, max_groups=None, device="cpu",
                                              static_order_dir=None, responder_source_dir=None)
        for extra in (["--split", "test"], ["--methods", "invalid"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(args + extra)

    def test_evaluation_static_artifact_dispatch(self):
        with patch("cbmjev.crossfit_evaluation.evaluate_crossfit_validation", return_value={}) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["evaluate-crossfit", "--prepared", "p", "--planned", "f", "--merged", "m",
                      "--responder", "r", "--cache", "c", "--out", "o",
                      "--methods", "static", "static_value", "--static-order-dir", "s"])
            self.assertEqual(evaluate.call_args.kwargs["static_order_dir"], "s")
            self.assertEqual(evaluate.call_args.kwargs["methods"], ["static", "static_value"])
            self.assertIsNone(evaluate.call_args.kwargs["responder_source_dir"])

    def test_static_dispatch_preserves_fold_sources(self):
        with patch("cbmjev.crossfit_static.fit_crossfit_static_order", return_value=([0], {})) as fit:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["fit-crossfit-static", "--prepared", "p", "--planned", "f",
                    "--outer-dirs", "o0", "o1", "o2", "--out", "order"]), 0)
            fit.assert_called_once_with("p", "f", ["o0", "o1", "o2"], "order",
                                        device="cpu", batch_size=64, responder_source_dir=None)

    def test_merge_dispatches_all_outer_paths_and_head_flag(self):
        for extra, expected in (([], True), (["--controller-only"], False)):
            with patch("cbmjev.crossfit_merge.merge_outer_folds", return_value={}) as merge:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["merge-crossfit-outer", "--prepared", "p",
                        "--planned", "f", "--outer-dirs", "o0", "o1", "o2",
                        "--out", "merged"] + extra), 0)
                merge.assert_called_once_with("p", "f", ["o0", "o1", "o2"], "merged",
                                              fit_final_head=expected, responder_source_dir=None,
                                              controller_batch_size=None)

    def test_historical_source_bridge_dispatch(self):
        with patch("cbmjev.crossfit_merge.merge_outer_folds", return_value={}) as merge:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["merge-crossfit-outer", "--prepared", "p",
                    "--planned", "f", "--outer-dirs", "o0", "o1", "o2", "--out", "merged",
                    "--responder-source-dir", "release", "--controller-batch-size", "4096"]), 0)
            self.assertEqual(merge.call_args.kwargs["responder_source_dir"], "release")
            self.assertEqual(merge.call_args.kwargs["controller_batch_size"], 4096)

    def test_train_cache_and_readback(self):
        from cbmjev.crossfit_cache import load_crossfit_cache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = fixture(root)
            audit = read_json(prepared / "audit.json")
            audit["source_revision"] = "SYNTHETIC_CLI_FIXTURE"
            write_json(prepared / "audit.json", audit)
            planned, responder, cache = root / "plan", root / "responder", root / "cache"
            plan_crossfit_prepared(prepared, planned, inner_folds=2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "train-crossfit-responder", "--prepared", str(prepared),
                    "--planned", str(planned), "--out", str(responder),
                    "--outer-fold", "0", "--epochs", "1", "--batch-size", "4",
                    "--input-workers", "2", "--input-prefetch-batches", "2"]), 0)
                self.assertEqual(main([
                    "cache-crossfit", "--prepared", str(prepared), "--planned", str(planned),
                    "--responder", str(responder), "--out", str(cache)]), 0)
            schema, rows, manifest = load_crossfit_cache(cache, prepared, planned, responder)
            self.assertEqual(len(rows), 4)
            self.assertEqual({r["group_id"] for r in rows}, {"g00", "g03", "g06", "g09"})
            self.assertEqual(schema.num_atoms, 1)

    def test_test_access_not_exposed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["cache-crossfit", "--prepared", "p", "--planned", "f",
                                 "--responder", "r", "--out", "o", "--split", "test"])

    def test_training_stage_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["train-crossfit-responder", "--prepared", "p",
                                 "--planned", "f", "--out", "o"])

    def test_legacy_defaults_preserved(self):
        args = parser().parse_args(["train-responder", "--prepared", "p", "--out", "o"])
        self.assertEqual((args.kind, args.epochs, args.device), ("hashing_text", 30, "cpu"))


if __name__ == "__main__":
    unittest.main()
