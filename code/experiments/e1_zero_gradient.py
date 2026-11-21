#!/usr/bin/env python3
"""E1: does correction remove structural silence (zero gradient)?

On an all-fail group (every on-policy rollout R=0), GRPO's group-relative
baseline makes the advantage identically zero, so its gradient is zero. VCPO
adds a certificate connection with R=1, so its gradient is non-zero.

This script measures the two gradient norms directly:
  - |grad GRPO| on an all-fail group  -> should be ~0 (structural silence)
  - |grad VCPO| (with the certificate) -> should be > 0 (correction works)
"""
import argparse
import numpy as np
import torch

import vcpo_common as common
from vcpo_common import load_model, encode_with_boundary, cosine, flat_grad
from e6_alignment import token_logprobs, grad_of


def greedy_rollout(model, tokenizer, prompt, device, max_new=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def grad_grpo(model, tokenizer, batch, device):
    """GRPO gradient on the on-policy rollouts. All R=0 -> advantage 0 -> grad 0."""
    rewards = [item["reward"] for item in batch]
    mean_r = sum(rewards) / len(rewards)
    var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    total = 0.0
    for item in batch:
        adv = (item["reward"] - mean_r) / std if std > 0 else 0.0
        ids, start = encode_with_boundary(tokenizer, item["prompt"], item["rollout"])
        total = total + adv * token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -total)


def grad_vcpo_conn(model, tokenizer, batch, device):
    """VCPO gradient from the certificate connection (R=1), detached coefficient."""
    log_coeffs = []
    entries = []
    for item in batch:
        ids, start = encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(len(batch)) + log_p, 0.0))
        log_coeffs.append(log_p - log_D)  # R=1 for the certificate
        entries.append((ids, start))
    max_log_c = max(log_coeffs)
    total = 0.0
    for (ids, start), log_c in zip(entries, log_coeffs):
        coeff = float(np.exp(log_c - max_log_c))
        total = total + coeff * token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-tasks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    common.set_seed(args.seed)
    model, tokenizer = load_model(args.model, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    from datasets import load_dataset
    rows = list(load_dataset("openai/openai_humaneval", split="test"))[:args.n_tasks]

    batch = []
    for row in rows:
        rollout = greedy_rollout(model, tokenizer, row["prompt"], device)
        batch.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "rollout": rollout,                      # on-policy, R=0 (all-fail)
            "certificate": row["canonical_solution"],  # R=1 connection
            "reward": 0.0,                            # greedy almost never solves
        })

    g_grpo = grad_grpo(model, tokenizer, batch, device).cpu()
    g_vcpo = grad_vcpo_conn(model, tokenizer, batch, device).cpu()
    torch.cuda.empty_cache()

    n_grpo = float(g_grpo.norm())
    n_vcpo = float(g_vcpo.norm())
    ratio = n_vcpo / n_grpo if n_grpo > 0 else float("inf")

    meta = common.RunMeta(
        experiment="e1_zero_gradient",
        model=args.model,
        dataset="humaneval",
        seed=args.seed,
        dtype="bfloat16",
        extra={"n_tasks": len(batch), "all_fail": True,
               "note": "on-policy greedy rollouts all marked R=0 (all-fail group)"},
    ).fill_environment()
    payload = {
        "grad_grpo_norm": n_grpo,
        "grad_vcpo_norm": n_vcpo,
        "norm_ratio": ratio,
    }
    path = common.write_result(f"e1_{args.model.split('/')[-1]}", meta, payload)

    print(f"model={args.model}  all-fail group (n={len(batch)})")
    print(f"  |grad GRPO| = {n_grpo:.6e}   (want ~0: structural silence)")
    print(f"  |grad VCPO| = {n_vcpo:.6e}   (want >0: correction works)")
    print(f"  ratio = {ratio:.3e}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
