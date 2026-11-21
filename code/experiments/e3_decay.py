#!/usr/bin/env python3
"""E3: how fast the corrected coefficient decays with completion length.

Lemma 1 gives c ~ pi_theta(certificate completion | prefix), a product of
per-token probabilities. Writing log10(p_bar) for the mean of the per-token
log-probabilities makes log10(c) = L * log10(p_bar) an *identity*, not a
prediction. The empirical questions this script answers are:

  1. Is p_bar stable enough across lengths and tasks for a single slope to fit?
     (Reported as R^2. A poor fit is informative, not a failed run.)
  2. How does the fitted slope vary with model and with task?

Two length units are reported. Token counts are not comparable across
tokenizer families -- the same solution is ~100 tokens under a 50K vocabulary
and ~70 under a 256K one -- so the cross-family table must use the
per-character slope. Both are written out.

Usage:
    python e3_decay.py --model Qwen/Qwen3-0.6B-Base --dataset humaneval
"""

from __future__ import annotations

import argparse
import json

import vcpo_common as common  # noqa: F401  (pins HF cache before transformers)
from vcpo_common import RunMeta, fit_line, load_model, sequence_logprob, write_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(common.LOADERS))
    parser.add_argument("--limit", type=int, default=None, help="cap tasks, for smoke runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    common.set_seed(args.seed)
    items = common.load_dataset_items(args.dataset, args.limit)
    model, tokenizer = load_model(args.model)

    points = []
    for item in items:
        try:
            measurement = sequence_logprob(model, tokenizer, item["prompt"], item["certificate"])
        except (ValueError, RuntimeError) as error:
            # Record rather than drop: a systematically skipped subset would
            # bias the fit, so the count of failures is part of the result.
            points.append({"task_id": item["task_id"], "error": str(error)})
            continue
        measurement.pop("token_log_probs")  # kept out of the summary file for size
        measurement["task_id"] = item["task_id"]
        points.append(measurement)

    usable = [p for p in points if "error" not in p]
    by_token = fit_line([p["n_tokens"] for p in usable], [p["log10_c"] for p in usable])
    by_char = fit_line([p["n_chars"] for p in usable], [p["log10_c"] for p in usable])

    lengths = sorted(p["n_tokens"] for p in usable)
    meta = RunMeta(
        experiment="e3_decay",
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        dtype="bfloat16",
        extra={
            "n_tasks": len(items),
            "n_usable": len(usable),
            "n_failed": len(points) - len(usable),
            "token_length_range": [lengths[0], lengths[-1]] if lengths else None,
            "quantity": "pi_theta(certificate completion | prompt), teacher-forced",
            "note": "log10(c)=L*log10(p_bar) is an identity; R2 tests whether one slope fits",
        },
    ).fill_environment()

    name = f"e3_{args.model.split('/')[-1]}_{args.dataset}{args.tag}"
    path = write_result(name, meta, {"fit_per_token": by_token, "fit_per_char": by_char, "points": points})

    print(f"model={args.model} dataset={args.dataset} n={len(usable)}/{len(items)}")
    print(f"  per-token: slope={by_token['slope']:+.4f} +/- {by_token['slope_stderr']:.4f}  R2={by_token['r2']:.3f}")
    print(f"  per-char : slope={by_char['slope']:+.4f} +/- {by_char['slope_stderr']:.4f}  R2={by_char['r2']:.3f}")
    if lengths:
        print(f"  L range  : {lengths[0]}-{lengths[-1]} tokens")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
