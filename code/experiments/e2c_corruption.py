#!/usr/bin/env python3
"""E2(c): certificate corruption sensitivity.

Does the ratio actually matter? Inject WRONG certificates (each task gets a
different task's solution) and measure how much the gradient direction changes.

  - VCPO (corrected, has ratio): p_theta(wrong_cert | prompt) ~ 0, so the wrong
    cert is auto-downweighted -> cos(clean, corrupted) should be ~ 1 (robust).
  - F_q  (no-ratio, reward-weighted log-likelihood): wrong cert gets full
    weight -> cos(clean, corrupted) should drop (sensitive).

This is the direct evidence that the ratio has a real effect, not just that it
is theoretically correct.
"""
import argparse
import torch

import vcpo_common as common
from vcpo_common import cosine, load_model, write_result, RunMeta
from e6_alignment import (
    build_batch, grad_vcpo, grad_F_q,
)


def corrupt(batch: list[dict]) -> list[dict]:
    """Cyclically shift certificates so each task gets a different task's solution."""
    certs = [item["certificate"] for item in batch]
    shifted = certs[1:] + certs[:1]
    out = []
    for item, wrong in zip(batch, shifted):
        d = dict(item)
        d["certificate"] = wrong
        if d["reward"] == 1.0:
            d["rollout"] = wrong  # R=1 rollout was the (now wrong) certificate
        out.append(d)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="humaneval", choices=sorted(common.LOADERS))
    parser.add_argument("--n-tasks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common.set_seed(args.seed)
    model, tokenizer = load_model(args.model, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device
    batch = build_batch(model, tokenizer, args.n_tasks, device, dataset=args.dataset)
    corrupted = corrupt(batch)

    # Compute cosines incrementally and free each gradient as soon as it is
    # used. A 14B flat gradient is ~28 GB in bf16; holding all four at once is
    # ~112 GB, which trips the container's ~120 GB cgroup RAM cap (exit 137).
    # At most two gradients are alive at any moment, so the peak stays ~84 GB.
    g_vcpo_clean = grad_vcpo(model, tokenizer, batch, device)
    g_vcpo_corrupted = grad_vcpo(model, tokenizer, corrupted, device)
    cos_vcpo = cosine(g_vcpo_clean, g_vcpo_corrupted)
    del g_vcpo_clean, g_vcpo_corrupted

    g_fq_clean = grad_F_q(model, tokenizer, batch, device)
    g_fq_corrupted = grad_F_q(model, tokenizer, corrupted, device)
    cos_fq = cosine(g_fq_clean, g_fq_corrupted)
    del g_fq_clean, g_fq_corrupted
    torch.cuda.empty_cache()

    result = {
        "cos_vcpo_clean_vs_corrupted": cos_vcpo,
        "cos_fq_clean_vs_corrupted": cos_fq,
    }

    meta = RunMeta(
        experiment="e2c_corruption",
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        dtype="bfloat16",
        extra={
            "n_tasks": len(batch),
            "corruption": "cyclic certificate swap (each task gets another task's solution)",
            "prediction": "VCPO robust (cos~1), F_q sensitive (cos drops)",
        },
    ).fill_environment()

    dsuffix = "" if args.dataset == "humaneval" else f"_{args.dataset}"
    path = write_result(f"e2c_{args.model.split('/')[-1]}{dsuffix}", meta, result)
    print(f"model={args.model}")
    print(f"  cos(vcpo clean, corrupted) = {result['cos_vcpo_clean_vs_corrupted']:.4f}  (want ~1: robust)")
    print(f"  cos(F_q  clean, corrupted) = {result['cos_fq_clean_vs_corrupted']:.4f}  (want <<1: sensitive)")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
