#!/usr/bin/env python3
"""Compare frozen value predictions with zero and in-sample state/action means."""
import argparse
from collections import defaultdict
from itertools import islice
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.io import write_json, file_hash
from cbmjev.learning import action_targets, policy_examples, training_rows
from cbmjev.pipeline import load_model_bundle, code_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-examples", type=int, default=20000)
    args = p.parse_args()
    if Path(args.out).exists() or args.max_examples < 1:
        raise ValueError("new output and positive max-examples required")
    schema, rows, manifest, cfg, receipt, head, controller, _ = load_model_bundle(
        args.models, args.cache, args.device)
    if controller.objective != "value":
        raise ValueError("value checkpoint required")
    results = {}
    for split in ("policy_fit", "validation"):
        selected = training_rows(rows, split, schema)
        iterator = islice(policy_examples(selected, schema, controller.config, epoch=0), args.max_examples)
        groups = defaultdict(lambda: [0, 0.0, 0.0])
        n = 0
        error = square = total = 0.0
        while True:
            examples = list(islice(iterator, 256))
            if not examples:
                break
            targets = action_targets(head, examples, "value").cpu().tolist()
            # Controller's public interface scores one observed state with many
            # actions. Group the batch to avoid one forward per training record.
            by_state = defaultdict(list)
            for i, (before, action, _, _) in enumerate(examples):
                by_state[before].append((i, action))
            predictions = [None] * len(examples)
            for before, entries in by_state.items():
                values = controller.predict(before, tuple(action for _, action in entries))
                for (i, _), value in zip(entries, values):
                    predictions[i] = value
            for example, target, pred in zip(examples, targets, predictions):
                before, action, _, _ = example
                if not action:
                    continue  # deterministic zero STOP must not dilute fit metrics
                n += 1
                error += (pred - target) ** 2
                square += target ** 2
                total += target
                group = groups[(before, action)]
                group[0] += 1
                group[1] += target
                group[2] += target ** 2
        floor = sum(max(0.0, q - s * s / count) for count, s, q in groups.values()) / n
        results[split] = {"nonstop_examples": n, "state_action_buckets": len(groups),
                          "singleton_buckets": sum(v[0] == 1 for v in groups.values()),
                          "controller_mse": error / n, "zero_mse": square / n,
                          "global_constant_insample_mse": square / n - (total / n) ** 2,
                          "state_action_mean_insample_mse": floor,
                          "mean_target": total / n,
                          "scope": "EMPIRICAL_INSAMPLE_BUCKET_MEAN_IS_OPTIMISTIC_NOT_BAYES_OR_DEPLOYABLE_BASELINE"}
    write_json(args.out, {"scope": "DEVELOPMENT_FIT_DIAGNOSTIC_NOT_TEST_OR_PAPER_EVIDENCE",
                         "results": results, "max_examples_including_stop": args.max_examples,
                         "models_sha256": receipt["models_sha256"],
                         "cache_responses_sha256": manifest["responses_sha256"],
                         "training_report_sha256": file_hash(Path(args.models) / "training.json"),
                         "evaluation_source_code_hash": code_fingerprint()})
    print(results)


if __name__ == "__main__":
    main()
