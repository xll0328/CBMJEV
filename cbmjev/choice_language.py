"""Frozen language features with paired Choice heads, not full-body NanoJev.

This low-level bridge binds supplied examples to encoding and training reports;
it does not authenticate their split, upstream targets, or checkpoint provenance.
"""
import torch

from .choice_encoding import _frozen_eval, encode_frozen_choice_features
from .choice_targets import ChoiceExample, ChoiceInputs
from .choice_training import fit_choice_head_pair
from .contracts import stable_hash
from .nano_choice import NanoChoiceHead


def _scorer_device(scorer):
    _frozen_eval(scorer)
    devices = {p.device for p in scorer.parameters()}
    if len(devices) != 1:
        raise ValueError("language scorer parameters must share one device")
    return next(iter(devices))


def _limits(max_padded_tokens, max_candidates_per_batch):
    if any(type(x) is not int or x < 1 for x in
           (max_padded_tokens, max_candidates_per_batch)):
        raise ValueError("explicit positive integer encoding limits required")


def fit_frozen_language_choice_pair(scorer, examples, schema, *, max_questions,
        max_padded_tokens, max_candidates_per_batch, epochs, batch_size, lr,
        seed, device="cpu"):
    """Encode a bounded example tuple once, then fit both heads on raw features.

    Backbone and head-training devices may differ explicitly: the encoder
    returns detached CPU tensors, and the trainer owns transfer to ``device``.
    Neither supervision nor source metadata reaches the language model.
    """
    encoding_device = _scorer_device(scorer)
    if (type(examples) is not tuple or not examples
            or any(not isinstance(e, ChoiceExample) for e in examples)):
        raise ValueError("nonempty tuple of ChoiceExample required")
    if any(not isinstance(e.model_inputs, ChoiceInputs)
           or any(not isinstance(a, tuple) or len(a) > 1 for a in e.model_inputs.actions)
           for e in examples):
        raise ValueError("language Choice training requires tuple singleton actions")
    features, encoder = encode_frozen_choice_features(scorer,
        tuple(e.model_inputs for e in examples), schema, max_questions=max_questions,
        max_padded_tokens=max_padded_tokens,
        max_candidates_per_batch=max_candidates_per_batch)
    scalar, attention, trainer = fit_choice_head_pair(examples, features,
        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, device=device)
    report = dict(format="cbmjev-frozen-language-choice-pair-v1",
        backend="frozen-language-choice", full_body_training=False,
        upstream_provenance_verified=False,
        provenance_scope="Content binding only; caller must verify upstream splits, targets and backbone.",
        objective="one-step soft Choice imitation, not calibrated risk",
        encoding_device=str(encoding_device), training_device=str(next(scalar.parameters()).device),
        feature_width=scorer.hidden_size, normalization_cached=False,
        encoder=encoder, trainer=trainer)
    report["binding_sha256"] = stable_hash(report)
    return scalar, attention, report


class FrozenLanguageChoiceController:
    """Duck-compatible with run_episode(method='structured_choice').

    The runtime name is a dispatch compatibility detail; this backend uses
    language features. Only the latest encoding receipt is retained, never a
    growing state/feature cache. Preference logits already condition on costs.
    """
    objective = "choice"
    backend = "frozen-language-choice"

    def __init__(self, scorer, head, schema, *, max_padded_tokens,
                 max_candidates_per_batch):
        _limits(max_padded_tokens, max_candidates_per_batch)
        self.scorer, self.head, self.schema = scorer, head, schema
        self.max_padded_tokens = max_padded_tokens
        self.max_candidates_per_batch = max_candidates_per_batch
        self.last_stats = None
        self._validate()

    def _validate(self):
        _scorer_device(self.scorer)
        if (not isinstance(self.head, NanoChoiceHead)
                or self.head.hidden_size != self.scorer.hidden_size):
            raise ValueError("Choice head width must match language feature width")
        if (any(m.training for m in self.head.modules())
                or any(p.requires_grad for p in self.head.parameters())):
            raise ValueError("inference requires an eval-mode frozen Choice head")
        parameters = tuple(self.head.parameters())
        if len({p.device for p in parameters}) != 1 or any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Choice head requires one device and float32 parameters")

    @torch.no_grad()
    def predict_logits(self, observed, actions, *, remaining_groups, cost, cost_weight):
        self.last_stats = None
        self._validate()
        observed, actions = tuple(observed), tuple(actions)
        if any(not isinstance(a, tuple) or len(a) > 1 for a in actions):
            raise ValueError("language Choice protocol supports tuple singleton actions only")
        inputs = ChoiceInputs(observed, actions, remaining_groups,
            tuple(cost(observed, a) for a in actions), cost_weight)
        features, report = encode_frozen_choice_features(self.scorer, (inputs,), self.schema,
            max_questions=1, max_padded_tokens=self.max_padded_tokens,
            max_candidates_per_batch=self.max_candidates_per_batch)
        device = next(self.head.parameters()).device
        raw = features[0].to(device=device, dtype=torch.float32).unsqueeze(0)
        valid = torch.ones(raw.shape[:2], dtype=torch.bool, device=device)
        logits = self.head(raw, valid)[0]
        self.last_stats = dict(backend=self.backend, objective=self.objective,
            head_device=str(device), normalization_cached=False,
            upstream_provenance_verified=False, full_body_training=False,
            encoding=report)
        return tuple(logits.cpu().tolist())
