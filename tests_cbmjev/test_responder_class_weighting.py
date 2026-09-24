"""Training-only class balancing; synthetic checks, not accuracy evidence."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from cbmjev.contracts import Concept, ModelInput, QueryGroup, Schema
from cbmjev.responders import ConceptTrainingExample, fit_responder, validate_examples


class ConstantResponder(nn.Module):
    def __init__(self):
        super().__init__()
        self.schema = Schema("synthetic_engineering", 2,
            tuple(Concept(str(j), str(j), ("negative", "positive")) for j in range(3)),
            tuple(QueryGroup(str(j), (j,)) for j in range(3)))
        self.logits = nn.Parameter(torch.tensor([[1., -1.], [.3, -.3], [0., 0.]]))

    def forward(self, payloads):
        return tuple(row.unsqueeze(0).expand(len(payloads), -1) for row in self.logits)


def examples():
    return [ConceptTrainingExample(ModelInput(text=str(i)), values) for i, values in enumerate(
        [(0, 1, None), (0, None, None), (0, 1, None), (1, None, None)])]


class ResponderClassWeightingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.addCleanup(torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled())

    def test_train_counts_and_fixed_weights_include_missing_classes(self):
        model = ConstantResponder()
        _, report = fit_responder(model, examples(), epochs=1, batch_size=4,
                                  concept_class_weighting="inverse_frequency")
        self.assertEqual(report["concept_class_counts"], [[3, 1], [0, 2], [0, 0]])
        self.assertEqual(report["concept_class_weights"], [[2 / 3, 2.], [0., 1.], [0., 0.]])
        self.assertEqual(report["observed_labels_per_atom"], [4, 2, 0])
        self.assertEqual(report["concept_class_weight_fit_scope"], "supplied_responder_fit_examples_only")
        self.assertFalse(report["task_label_gradient"])
        for counts, weights in zip(report["concept_class_counts"], report["concept_class_weights"]):
            if sum(counts):
                self.assertAlmostEqual(sum(c*w for c, w in zip(counts, weights))/sum(counts), 1.)

    def test_weighted_loss_matches_fixed_population_formula(self):
        model = ConstantResponder()
        ce = F.cross_entropy(model.logits[0].expand(4, -1), torch.tensor([0, 0, 0, 1]), reduction="none")
        expected = ((ce * torch.tensor([2/3, 2/3, 2/3, 2.])).mean() +
                    F.cross_entropy(model.logits[1:2], torch.tensor([1]))) / 2
        _, report = fit_responder(model, examples(), epochs=1, batch_size=4,
                                  concept_class_weighting="inverse_frequency")
        self.assertAlmostEqual(report["training_loss"][0], float(expected.detach()), places=6)

    def test_validation_and_test_rows_rejected_not_counted(self):
        for split in ("validation", "test", "policy_fit"):
            model = ConstantResponder()
            initial = model.logits.detach().clone()
            contaminant = ConceptTrainingExample(ModelInput(text="heldout"), (1, 0, 1), split)
            with self.assertRaisesRegex(ValueError, "responder_fit only"):
                fit_responder(model, examples() + [contaminant], epochs=1,
                              concept_class_weighting="inverse_frequency")
            self.assertTrue(torch.equal(initial, model.logits))

    def test_singleton_batches_do_not_cancel_rare_class_weight(self):
        model = ConstantResponder()
        rows = [ConceptTrainingExample(ModelInput(text=str(i)), (value, None, None))
                for i, value in enumerate([0, 0, 0, 1])]
        # Hold parameters fixed to inspect the exact four minibatch losses.
        ce = F.cross_entropy(model.logits[0].expand(4, -1), torch.tensor([0, 0, 0, 1]), reduction="none")
        expected = float((ce * torch.tensor([2/3, 2/3, 2/3, 2.])).mean().detach())
        with patch.object(torch.optim.Adam, "step", return_value=None):
            _, report = fit_responder(model, rows, epochs=1, batch_size=1,
                                      concept_class_weighting="inverse_frequency")
        self.assertAlmostEqual(report["training_loss"][0], expected, places=6)

    def test_validation_single_pass_and_legacy_return(self):
        class CountReads(list):
            passes = 0
            def __iter__(self):
                self.passes += 1
                return super().__iter__()
        rows = CountReads(examples())
        counts, classes = validate_examples(rows, ConstantResponder().schema, return_class_counts=True)
        self.assertEqual(rows.passes, 1)
        self.assertEqual(counts, [4, 2, 0])
        self.assertEqual(classes, [[3, 1], [0, 2], [0, 0]])
        self.assertEqual(validate_examples(examples(), ConstantResponder().schema), counts)

    def test_none_exact_legacy_training_parity(self):
        # Independent legacy loop, including minibatch shuffle and equal-concept CE.
        original = ConstantResponder()
        reference = copy.deepcopy(original)
        rows = examples()
        optimizer = torch.optim.Adam(reference.parameters(), lr=.003)
        generator = torch.Generator().manual_seed(23)
        legacy_losses = []
        for _ in range(2):
            losses = []
            for indices in torch.randperm(len(rows), generator=generator).split(2):
                batch = [rows[i] for i in indices]
                terms = []
                for j, pred in enumerate(reference([e.payload for e in batch])):
                    valid = [i for i, e in enumerate(batch) if e.concepts[j] is not None]
                    if valid:
                        terms.append(F.cross_entropy(pred[valid], torch.tensor([batch[i].concepts[j] for i in valid])))
                loss = torch.stack(terms).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            legacy_losses.append(sum(losses) / len(losses))
        default, report = fit_responder(copy.deepcopy(original), rows, epochs=2, batch_size=2, seed=23)
        explicit, report2 = fit_responder(copy.deepcopy(original), rows, epochs=2, batch_size=2, seed=23,
                                         concept_class_weighting="none")
        self.assertTrue(torch.equal(default.logits, reference.logits))
        self.assertTrue(torch.equal(explicit.logits, reference.logits))
        self.assertEqual(report["training_loss"], legacy_losses)
        self.assertEqual(report, report2)
        self.assertEqual(report["concept_class_weights"], [[1., 1.]] * 3)

    def test_unknown_mode_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "concept_class_weighting"):
            fit_responder(ConstantResponder(), examples(), concept_class_weighting="validation_frequency")


if __name__ == "__main__":
    unittest.main()
