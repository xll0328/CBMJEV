"""CPU learners for the hard-concept, single-verification protocol.

Fit/tune arrays are supplied explicitly: this module never splits data, reads a
dataset, or sees confirmation data. All semantic categories are zero based.
The task head receives hard fused concepts; selectors receive hard initial A
only. One-hot encoding happens inside the models. Offline B responses and task
labels are supervision, never predictor arguments or retained training data.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _counts(category_counts):
    values = tuple(category_counts)
    if not values:
        raise ValueError("category_counts must not be empty")
    return tuple(_integer(c, "category count", minimum=2) for c in values)


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _hard_matrix(values, counts, name, *, nonempty=True):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(counts) or (nonempty and not len(array)):
        raise ValueError(f"{name} must have shape (samples, {len(counts)})" +
                         (" with at least one sample" if nonempty else ""))
    if array.dtype.kind == "O":
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in array.flat):
            raise ValueError(f"{name} must contain hard integer category IDs")
    elif array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain hard integer category IDs")
    for j, count in enumerate(counts):
        if np.any(array[:, j] < 0) or np.any(array[:, j] >= count):
            raise ValueError(f"{name} category {j} must be in [0, {count})")
    return torch.tensor(array.astype(np.int64, copy=False), dtype=torch.long)


def _labels(values, n, classes, name):
    array = np.asarray(values)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    return _hard_matrix(array[:, None], (classes,), name).flatten()


def _weights(values, n, name):
    if values is None:
        return torch.ones(n, dtype=torch.float32)
    array = np.asarray(values)
    if array.shape != (n,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector of length {n}")
    if not np.isfinite(array).all() or np.any(array < 0) or not np.isfinite(array.sum()) or array.sum() <= 0:
        raise ValueError(f"{name} must be finite, nonnegative, and have positive total weight")
    # Only relative weights matter; rescaling also avoids float32 overflow.
    scaled = array.astype(np.float64) / float(array.max())
    result = torch.tensor(scaled, dtype=torch.float32)
    if not torch.isfinite(result).all() or result.sum() <= 0:
        raise ValueError(f"{name} cannot be represented as finite positive-total weights")
    return result


def _signed_gains(values, n, k, name):
    array = np.asarray(values)
    if array.shape != (n, k) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have shape ({n}, {k}) and contain signed accuracy gains")
    if not np.isfinite(array).all() or not np.isin(array, (-1, 0, 1)).all():
        raise ValueError(f"{name} must contain only -1, 0, 1 (damage, unchanged, repair)")
    return torch.tensor(array, dtype=torch.float32)


def _one_hot(concepts, counts):
    if (not isinstance(concepts, torch.Tensor) or concepts.ndim != 2
            or concepts.shape[1] != len(counts) or concepts.device.type != "cpu"
            or concepts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("model inputs must be a CPU hard-integer (samples, concepts) tensor")
    if any(torch.any(concepts[:, j] < 0) or torch.any(concepts[:, j] >= c)
           for j, c in enumerate(counts)):
        raise ValueError("model input category out of range")
    return torch.cat([F.one_hot(concepts[:, j].long(), c) for j, c in enumerate(counts)], dim=1).float()


class TaskMLP(nn.Module):
    """Shared task f: one-hot hard fused concepts -> task logits."""

    def __init__(self, category_counts, num_classes, *, hidden=128, dropout=0.1):
        super().__init__()
        self.category_counts = _counts(category_counts)
        self.num_classes = _integer(num_classes, "num_classes", minimum=2)
        self.hidden = _integer(hidden, "hidden")
        self.dropout = _number(dropout, "dropout")
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1")
        self.network = nn.Sequential(nn.Linear(sum(self.category_counts), self.hidden),
                                     nn.ReLU(), nn.Dropout(self.dropout),
                                     nn.Linear(self.hidden, self.num_classes))

    def forward(self, concepts):
        return self.network(_one_hot(concepts, self.category_counts))


class SignedGainMLP(nn.Module):
    """Initial A only -> unconstrained signed-gain estimates per concept."""

    def __init__(self, category_counts, *, hidden=64):
        super().__init__()
        self.category_counts = _counts(category_counts)
        self.hidden = _integer(hidden, "hidden")
        self.network = nn.Sequential(nn.Linear(sum(self.category_counts), self.hidden),
                                     nn.ReLU(), nn.Linear(self.hidden, len(self.category_counts)))

    def forward(self, initial_a):
        return self.network(_one_hot(initial_a, self.category_counts))


class JointResponseMLP(nn.Module):
    """Shared qY and conditional or separately fitted factorized responses.

    Conditional heads have distinct weights for every (Y, response) pair.
    Consequently Y changes the weights multiplying h, not merely an additive
    intercept: r_j can represent h-by-Y interactions. A forward pass enumerates
    all Y; gold Y is used solely to index response CE during fitting.
    """

    def __init__(self, category_counts, num_classes, *, hidden=64, factorized=False):
        super().__init__()
        self.category_counts = _counts(category_counts)
        self.num_classes = _integer(num_classes, "num_classes", minimum=2)
        self.hidden = _integer(hidden, "hidden")
        if not isinstance(factorized, bool):
            raise ValueError("factorized must be a boolean")
        self.factorized = factorized
        self.representation = nn.Sequential(nn.Linear(sum(self.category_counts), self.hidden), nn.ReLU())
        self.y_head = nn.Linear(self.hidden, self.num_classes)
        rows = 1 if factorized else self.num_classes
        self.response_heads = nn.ModuleList(nn.Linear(self.hidden, rows * c) for c in self.category_counts)

    def forward(self, initial_a):
        h = self.representation(_one_hot(initial_a, self.category_counts))
        rows = 1 if self.factorized else self.num_classes
        responses = tuple(head(h).reshape(len(h), rows, c)
                          for head, c in zip(self.response_heads, self.category_counts))
        return self.y_head(h), responses


@dataclass(frozen=True)
class FitConfig:
    learning_rate: float
    weight_decay: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "learning_rate", _number(self.learning_rate, "learning_rate", positive=True))
        object.__setattr__(self, "weight_decay", _number(self.weight_decay, "weight_decay"))


TASK_CONFIGS = (FitConfig(3e-4), FitConfig(1e-3))
SELECTOR_CONFIGS = tuple(FitConfig(lr, wd) for lr in (3e-4, 1e-3) for wd in (0.0, 1e-3))


def _settings(*, seed, epochs, configs, batch_size, cpu_threads, hidden, synthetic,
              default_epochs, default_configs, default_hidden):
    seed = _integer(seed, "seed", minimum=0)
    if seed > 2**63 - 1:
        raise ValueError("seed must be <= 2**63 - 1")
    epochs = _integer(epochs, "epochs")
    batch_size = _integer(batch_size, "batch_size")
    cpu_threads = _integer(cpu_threads, "cpu_threads")
    hidden = _integer(hidden, "hidden")
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be a boolean")
    if epochs > default_epochs:
        raise ValueError(f"epochs exceed the fixed {default_epochs}-epoch budget")
    if configs is None:
        configs = default_configs
    else:
        configs = tuple(config if isinstance(config, FitConfig) else
                        FitConfig(**config) if isinstance(config, dict) else FitConfig(*config)
                        for config in configs)
    if not configs or len(configs) > len(default_configs) or len(set(configs)) != len(configs):
        raise ValueError("configs must be nonempty, unique, and within the search budget")
    if not synthetic and (epochs != default_epochs or configs != default_configs or hidden != default_hidden):
        raise ValueError("epoch, config, or width overrides require synthetic=True")
    return seed, epochs, configs, batch_size, cpu_threads, hidden


@contextmanager
def _cpu_context(threads):
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)
            torch.set_num_threads(previous_threads)


def _objective(model, x, targets, kind):
    if kind == "task":
        return F.cross_entropy(model(x), targets[0], reduction="none")
    if kind == "gain":
        return F.mse_loss(model(x), targets[0], reduction="none").mean(dim=1)
    y, b = targets
    y_logits, response_logits = model(x)
    loss_y = F.cross_entropy(y_logits, y, reduction="none")
    indices = torch.arange(len(x))
    response_loss = torch.stack([
        F.cross_entropy(logits[:, 0] if model.factorized else logits[indices, y],
                        b[:, j], reduction="none")
        for j, logits in enumerate(response_logits)], dim=1).mean(dim=1)
    return loss_y + response_loss


def _mean_objective(model, x, targets, weights, kind, batch_size):
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            sl = slice(start, start + batch_size)
            loss = _objective(model, x[sl], tuple(t[sl] for t in targets), kind)
            total += float((loss.double() * weights[sl].double()).sum())
    return total / float(weights.double().sum())


def _train_epoch(model, optimizer, x, targets, weights, kind, generator, batch_size):
    model.train()
    weight_sum = weights.double().sum().item()
    gradient_scale = len(x) / (min(batch_size, len(x)) * weight_sum)
    total = 0.0
    for indices in torch.randperm(len(x), generator=generator).split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        losses = _objective(model, x[indices], tuple(t[indices] for t in targets), kind)
        objective = (losses * weights[indices]).sum() * gradient_scale
        if not torch.isfinite(objective):
            raise FloatingPointError("non-finite training objective")
        objective.backward()
        optimizer.step()
        total += float((losses.detach().double() * weights[indices].double()).sum())
    model.eval()
    return total / weight_sum


def _fit(model_factory, train_x, train_targets, tune_x, tune_targets, *, train_weights,
         tune_weights, kind, settings, synthetic):
    seed, epochs, configs, batch_size, cpu_threads, hidden = settings
    started = perf_counter()
    trials, best_state, best_config, best_score = [], None, None, float("inf")
    best_epoch = None
    with _cpu_context(cpu_threads):
        for config in configs:
            # Identical initialization and permutations across configurations.
            torch.random.default_generator.manual_seed(seed)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            model = model_factory(hidden)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            trial_started = perf_counter()
            history, trial_best_score, trial_best_epoch = [], float("inf"), None
            for epoch in range(1, epochs + 1):
                train_loss = _train_epoch(model, optimizer, train_x, train_targets, train_weights,
                                          kind, generator, batch_size)
                score = _mean_objective(model, tune_x, tune_targets, tune_weights, kind, batch_size)
                if not np.isfinite(score):
                    raise FloatingPointError("non-finite tune objective")
                history.append({"epoch": epoch, "train_loss": train_loss, "tune_loss": score})
                if score < trial_best_score:
                    trial_best_score, trial_best_epoch = score, epoch
                if score < best_score:
                    best_score, best_config, best_epoch = score, config, epoch
                    best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            trials.append({"config": asdict(config), "best_epoch": trial_best_epoch,
                           "best_tune_loss": trial_best_score, "history": history,
                           "runtime_seconds": perf_counter() - trial_started})
        model = model_factory(hidden)
        model.load_state_dict(best_state)
        model.eval()
    metric = {"task": "weighted_task_nll", "gain": "concept_mean_signed_gain_mse",
              "joint": "task_nll_plus_concept_mean_conditional_response_nll",
              "factorized": "task_nll_plus_concept_mean_factorized_response_nll"}[kind]
    report = {
        "kind": kind, "seed": seed, "device": "cpu", "dtype": "float32", "cpu_threads": cpu_threads,
        "n_train": len(train_x), "n_tune": len(tune_x), "split_names": ["explicit_train", "explicit_tune"],
        "category_counts": list(model.category_counts), "hidden": hidden,
        "num_classes": getattr(model, "num_classes", None), "dropout": getattr(model, "dropout", 0.0),
        "parameter_count": sum(p.numel() for p in model.parameters()), "optimizer": "AdamW",
        "batch_size": batch_size, "epochs_per_config": epochs, "early_stopping": False,
        "synthetic_override": synthetic, "selection_metric": metric,
        "selected_config": asdict(best_config), "selected_epoch": best_epoch, "best_tune_loss": best_score,
        "tie_rule": "first config then earliest epoch", "trials": trials,
        "training_aggregation": "global weight normalization; partial last batch scaled proportionally",
        "tune_aggregation": "global weighted sample mean; response/gain coordinates equally averaged",
        "runtime_seconds": perf_counter() - started,
    }
    return model, report


@dataclass
class TaskHead:
    model: TaskMLP
    report: dict

    def predict_proba(self, concepts, *, batch_size=4096):
        x = _hard_matrix(concepts, self.model.category_counts, "concepts", nonempty=False)
        batch_size = _integer(batch_size, "batch_size")
        self.model.eval()
        with torch.inference_mode():
            parts = [self.model(batch).softmax(dim=1).numpy() for batch in x.split(batch_size)]
        return np.concatenate(parts, axis=0) if parts else np.empty((0, self.model.num_classes), dtype=np.float32)

    def save(self, path):
        save_model(self, path)


@dataclass
class GainSelector:
    model: SignedGainMLP
    report: dict

    def predict_gains(self, initial_a, *, batch_size=4096):
        x = _hard_matrix(initial_a, self.model.category_counts, "initial_a", nonempty=False)
        batch_size = _integer(batch_size, "batch_size")
        self.model.eval()
        with torch.inference_mode():
            parts = [self.model(batch).numpy() for batch in x.split(batch_size)]
        return np.concatenate(parts, axis=0) if parts else np.empty((0, len(self.model.category_counts)), dtype=np.float32)

    def save(self, path):
        save_model(self, path)


@dataclass
class JointSelector:
    model: JointResponseMLP
    report: dict

    def predict_joint(self, initial_a, *, batch_size=4096) -> Tuple[np.ndarray, ...]:
        """Return one normalized (N, Y, C_j) joint per concept, without gold Y."""
        x = _hard_matrix(initial_a, self.model.category_counts, "initial_a", nonempty=False)
        batch_size = _integer(batch_size, "batch_size")
        self.model.eval()
        parts = [[] for _ in self.model.category_counts]
        with torch.inference_mode():
            for batch in x.split(batch_size):
                y_logits, response_logits = self.model(batch)
                qy = y_logits.softmax(dim=1)
                for output, response in zip(parts, response_logits):
                    output.append((qy[:, :, None] * response.softmax(dim=2)).numpy())
        return tuple(np.concatenate(part, axis=0) if part else
                     np.empty((0, self.model.num_classes, c), dtype=np.float32)
                     for part, c in zip(parts, self.model.category_counts))

    def save(self, path):
        save_model(self, path)


def fit_task_head(train_concepts, train_y, tune_concepts, tune_y, *, category_counts,
                  num_classes, seed, train_weights=None, tune_weights=None, epochs=50,
                  configs=None, batch_size=256, cpu_threads=1, hidden=128, synthetic=False):
    """Fit shared f on explicitly supplied weighted hard fused states.

    The caller supplies per-state weights to implement its A-only/all-fused/
    singleton-mixture allocation. Train and tune NLL use global weight sums,
    never per-batch normalization. The full two-configuration, 50-epoch search
    is fixed unless explicitly marked synthetic.
    """
    counts, classes = _counts(category_counts), _integer(num_classes, "num_classes", minimum=2)
    settings = _settings(seed=seed, epochs=epochs, configs=configs, batch_size=batch_size,
                         cpu_threads=cpu_threads, hidden=hidden, synthetic=synthetic,
                         default_epochs=50, default_configs=TASK_CONFIGS, default_hidden=128)
    train_x = _hard_matrix(train_concepts, counts, "train_concepts")
    tune_x = _hard_matrix(tune_concepts, counts, "tune_concepts")
    train_targets = (_labels(train_y, len(train_x), classes, "train_y"),)
    tune_targets = (_labels(tune_y, len(tune_x), classes, "tune_y"),)
    model, report = _fit(lambda width: TaskMLP(counts, classes, hidden=width), train_x, train_targets,
                         tune_x, tune_targets, train_weights=_weights(train_weights, len(train_x), "train_weights"),
                         tune_weights=_weights(tune_weights, len(tune_x), "tune_weights"), kind="task",
                         settings=settings, synthetic=synthetic)
    return TaskHead(model, report)


def refit_task_head(all_concepts, all_y, *, selected, seed, weights=None,
                    batch_size=None, cpu_threads=1):
    """Refit f on all supplied OOF states at the frozen selected LR and epoch.

    This is a fresh initialization, not continuation from the selected weights.
    No tune data or metric is consulted. The final epoch is retained, and the
    original aggregate selection report is embedded for provenance.
    """
    if not isinstance(selected, TaskHead) or selected.report.get("selection_metric") != "weighted_task_nll":
        raise ValueError("selected must be a task head selected by weighted tune NLL")
    config = FitConfig(**selected.report["selected_config"])
    epochs = _integer(selected.report["selected_epoch"], "selected_epoch")
    seed = _integer(seed, "seed", minimum=0)
    if epochs > 50 or seed > 2**63 - 1:
        raise ValueError("invalid frozen epoch count or seed")
    batch_size = _integer(selected.report["batch_size"] if batch_size is None else batch_size, "batch_size")
    cpu_threads = _integer(cpu_threads, "cpu_threads")
    x = _hard_matrix(all_concepts, selected.model.category_counts, "all_concepts")
    targets = (_labels(all_y, len(x), selected.model.num_classes, "all_y"),)
    sample_weights = _weights(weights, len(x), "weights")
    started = perf_counter()
    history = []
    with _cpu_context(cpu_threads):
        torch.random.default_generator.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model = TaskMLP(selected.model.category_counts, selected.model.num_classes,
                        hidden=selected.model.hidden, dropout=selected.model.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        for epoch in range(1, epochs + 1):
            loss = _train_epoch(model, optimizer, x, targets, sample_weights, "task", generator, batch_size)
            history.append({"epoch": epoch, "train_loss": loss})
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise FloatingPointError("non-finite refitted task weights")
    report = {
        "kind": "task", "fit_mode": "refit_frozen_selection", "seed": seed,
        "device": "cpu", "dtype": "float32", "cpu_threads": cpu_threads,
        "n_train": len(x), "n_tune": 0, "split_names": ["explicit_refit"],
        "category_counts": list(model.category_counts), "hidden": model.hidden,
        "num_classes": model.num_classes, "dropout": model.dropout,
        "parameter_count": sum(p.numel() for p in model.parameters()), "optimizer": "AdamW",
        "batch_size": batch_size, "epochs": epochs, "early_stopping": False,
        "synthetic_override": selected.report["synthetic_override"],
        "selected_config": asdict(config), "selected_epoch": epochs,
        "selection_metric": "inherited_weighted_task_nll", "selection_report": deepcopy(selected.report),
        "checkpoint_rule": "final frozen epoch", "history": history,
        "training_aggregation": "global weight normalization; partial last batch scaled proportionally",
        "runtime_seconds": perf_counter() - started,
    }
    return TaskHead(model, report)


def fit_gain_selector(train_a, train_gains, tune_a, tune_gains, *, category_counts, seed,
                      epochs=100, configs=None, batch_size=256, cpu_threads=1, hidden=64, synthetic=False):
    """Fit raw signed {-1,0,1} task gains with MSE, retaining negative targets."""
    counts = _counts(category_counts)
    settings = _settings(seed=seed, epochs=epochs, configs=configs, batch_size=batch_size,
                         cpu_threads=cpu_threads, hidden=hidden, synthetic=synthetic,
                         default_epochs=100, default_configs=SELECTOR_CONFIGS, default_hidden=64)
    train_x = _hard_matrix(train_a, counts, "train_a")
    tune_x = _hard_matrix(tune_a, counts, "tune_a")
    train_targets = (_signed_gains(train_gains, len(train_x), len(counts), "train_gains"),)
    tune_targets = (_signed_gains(tune_gains, len(tune_x), len(counts), "tune_gains"),)
    model, report = _fit(lambda width: SignedGainMLP(counts, hidden=width), train_x, train_targets,
                         tune_x, tune_targets, train_weights=_weights(None, len(train_x), "train_weights"),
                         tune_weights=_weights(None, len(tune_x), "tune_weights"), kind="gain",
                         settings=settings, synthetic=synthetic)
    return GainSelector(model, report)


def fit_joint_selector(train_a, train_y, train_b, tune_a, tune_y, tune_b, *, category_counts,
                       num_classes, seed, factorized=False, epochs=100, configs=None,
                       batch_size=256, cpu_threads=1, hidden=64, synthetic=False):
    """Fit qY plus mean response CE; tune the same proper objective.

    With factorized=False, training indexes r_j(V|Y,h) at the training Y.
    With factorized=True, a fresh model fits r_j(V|h). Neither predictor takes
    B or Y at inference; qY is shared across every returned concept joint.
    """
    counts, classes = _counts(category_counts), _integer(num_classes, "num_classes", minimum=2)
    if not isinstance(factorized, bool):
        raise ValueError("factorized must be a boolean")
    settings = _settings(seed=seed, epochs=epochs, configs=configs, batch_size=batch_size,
                         cpu_threads=cpu_threads, hidden=hidden, synthetic=synthetic,
                         default_epochs=100, default_configs=SELECTOR_CONFIGS, default_hidden=64)
    train_x = _hard_matrix(train_a, counts, "train_a")
    tune_x = _hard_matrix(tune_a, counts, "tune_a")
    train_targets = (_labels(train_y, len(train_x), classes, "train_y"),
                     _hard_matrix(train_b, counts, "train_b"))
    tune_targets = (_labels(tune_y, len(tune_x), classes, "tune_y"),
                    _hard_matrix(tune_b, counts, "tune_b"))
    if len(train_targets[1]) != len(train_x) or len(tune_targets[1]) != len(tune_x):
        raise ValueError("B responses must have the same sample counts as their A and Y arrays")
    model, report = _fit(lambda width: JointResponseMLP(counts, classes, hidden=width, factorized=factorized),
                         train_x, train_targets, tune_x, tune_targets,
                         train_weights=_weights(None, len(train_x), "train_weights"),
                         tune_weights=_weights(None, len(tune_x), "tune_weights"),
                         kind="factorized" if factorized else "joint", settings=settings, synthetic=synthetic)
    return JointSelector(model, report)


def save_model(predictor, path):
    """Save weights, model config, and aggregate report, without fit/tune arrays."""
    if not isinstance(predictor, (TaskHead, GainSelector, JointSelector)):
        raise TypeError("predictor must be a TaskHead, GainSelector, or JointSelector")
    model = predictor.model
    config = {"category_counts": list(model.category_counts), "hidden": model.hidden}
    if isinstance(model, TaskMLP):
        kind = "task"
        config.update(num_classes=model.num_classes, dropout=model.dropout)
    elif isinstance(model, JointResponseMLP):
        kind = "joint"
        config.update(num_classes=model.num_classes, factorized=model.factorized)
    else:
        kind = "gain"
    torch.save({"format_version": 1, "kind": kind, "model_config": config,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "report": predictor.report}, path)


def load_model(path):
    """Load a weights-only checkpoint on CPU without changing the caller's RNG."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported verification learner checkpoint")
    constructors = {"task": (TaskMLP, TaskHead), "gain": (SignedGainMLP, GainSelector),
                    "joint": (JointResponseMLP, JointSelector)}
    if payload.get("kind") not in constructors:
        raise ValueError("unsupported verification learner kind")
    model_class, predictor_class = constructors[payload["kind"]]
    with torch.random.fork_rng(devices=[]):
        model = model_class(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("checkpoint contains non-finite weights")
    model.eval()
    return predictor_class(model, payload["report"])
