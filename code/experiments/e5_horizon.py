#!/usr/bin/env python3
"""E5: horizon scan — the connection coefficient collapses with length.

M3 claims the honest VCPO weight is c = p_bar^L and collapses to ~1e-20 for
realistic certificates. E3 fits the per-token slope across natural length
variation; E5 makes the collapse *direct* by holding the certificate fixed and
sweeping its prefix length L. For each task we compute the full per-token
log10-probs once, then take the cumulative sum to get log10(c) at every prefix
length. Averaging across tasks gives one collapse curve, and the mean slope of
that curve is log10(p_bar).
"""
import argparse
import numpy as np
import torch

import vcpo_common as common
from vcpo_common import load_model, encode_with_boundary, sequence_logprob, write_result, RunMeta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="humaneval", choices=sorted(common.LOADERS))
    parser.add_argument("--limit", type=int, default=None, help="cap tasks, for smoke runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common.set_seed(args.seed)
    model, tokenizer = load_model(args.model, train=False)
    device = next(model.parameters()).device

    items = [
        (item["prompt"], item["certificate"])
        for item in common.load_dataset_items(args.dataset, args.limit)
    ]

    # Per-task prefix curve: cumulative log10(c) at every token position.
    prefix_curves = []
    per_token_log10 = []
    for prompt, cert in items:
        info = sequence_logprob(model, tokenizer, prompt, cert)
        toks = info["token_log_probs"]  # natural-log per-token log-probs
        log10_toks = [t / np.log(10.0) for t in toks]
        per_token_log10.extend(log10_toks)
        prefix_curves.append(np.cumsum(log10_toks).tolist())

    # Align all curves to the same length axis (up to the min length).
    max_L = max(len(c) for c in prefix_curves)
    lengths = list(range(1, max_L + 1))
    mean_curve = []
    for L in lengths:
        vals = [c[L - 1] for c in prefix_curves if len(c) >= L]
        mean_curve.append(float(np.mean(vals)))

    # Fit log10(c) = L * log10(p_bar) over the range where enough tasks survive.
    report_Ls = [5, 10, 15, 20, 25, 30, 40, 50]
    report = {str(L): mean_curve[L - 1] for L in report_Ls if L <= max_L}
    fit_L = [L for L in lengths if L <= max_L]
    fit_y = [mean_curve[L - 1] for L in fit_L]
    slope = float(np.polyfit(fit_L, fit_y, 1)[0]) if len(fit_L) >= 3 else float("nan")

    meta = RunMeta(
        experiment="e5_horizon",
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        dtype="bfloat16",
        extra={"n_tasks": len(prefix_curves), "max_L": max_L,
               "metric": "log10(c) = sum of per-token log10-probs; slope = log10(p_bar)"},
    ).fill_environment()
    payload = {
        "report_lengths": report,
        "slope_log10_pbar": slope,
        "n_tasks": len(prefix_curves),
        "max_L": max_L,
        "per_token_log10_mean": float(np.mean(per_token_log10)),
    }
    path = write_result(f"e5_{args.model.split('/')[-1]}_{args.dataset}", meta, payload)
    print(f"model={args.model} dataset={args.dataset} n_tasks={len(prefix_curves)} max_L={max_L}")
    print(f"  log10(c) at L=5/10/20/30/40/50: " + "  ".join(
        f"{report.get(str(L), float('nan')):.2f}" for L in report_Ls))
    print(f"  slope (log10 p_bar) = {slope:.4f}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
