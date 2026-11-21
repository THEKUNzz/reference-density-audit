#!/usr/bin/env python3
"""Kernel 探针：原版 e2b 训练路径（CHUNK=4、checkpointing 开、全量前向）+ 已装 kernel。

与存档 ``e2b_Qwen3.5-9B_seed0.json`` 逐点对，只报 4 个数，不判定。
唯一变量是 kernel（flash-linear-attention + causal_conv1d），代码路径与存档那次完全一致。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import vcpo_common as common
from vcpo_common import load_model, set_seed
from e2b_overfit import (
    build_train_with_rewards, encode_batch, mean_logprob,
    train_step_fq, train_step_vcpo,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-heldout", type=int, default=40)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--n-rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--archive", default="data/results/e2b_Qwen3.5-9B_seed0.json")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, train=False)
    device = next(model.parameters()).device

    from datasets import load_dataset
    rows = list(load_dataset("openai/openai_humaneval", split="test"))
    train_mixed = build_train_with_rewards(model, tokenizer, rows, device,
                                           n_train=args.n_train, n_rollouts=args.n_rollouts)
    n_r1 = sum(1 for _, _, r in train_mixed if r > 0)
    n_r0 = sum(1 for _, _, r in train_mixed if r == 0)
    print(f"train batch: n_R1={n_r1} n_R0={n_r0}", flush=True)

    train_certs = [(r["prompt"], r["canonical_solution"]) for r in rows[: args.n_train]]
    heldout_certs = [(r["prompt"], r["canonical_solution"]) for r in rows[args.n_train : args.n_train + args.n_heldout]]
    train_enc = encode_batch(tokenizer, [it[:2] for it in train_mixed])
    rewards = [it[2] for it in train_mixed]
    train_cert_enc = encode_batch(tokenizer, train_certs)
    heldout_enc = encode_batch(tokenizer, heldout_certs)
    del model
    torch.cuda.empty_cache()

    # 原版训练模型：checkpointing 保持开（load_model(train=True) 默认开）
    model, _ = load_model(args.model, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device
    print(f"gradient_checkpointing = {model.is_gradient_checkpointing}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    curves = {}
    step_times = {}
    for name, step_fn in (("fq", train_step_fq), ("vcpo", train_step_vcpo)):
        series = []
        times = []
        for s in range(args.steps + 1):
            if s > 0:
                t0 = time.time()
                step_fn(model, *train_enc, rewards, device)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            if s % 10 == 0:
                tr = mean_logprob(model, train_cert_enc, device)
                ho = mean_logprob(model, heldout_enc, device)
                series.append({"step": s, "train_fit": tr, "heldout_fit": ho})
                print(f"  {name} step={s}: train={tr:.6f} heldout={ho:.6f}", flush=True)
        curves[name] = series
        step_times[name] = times
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\n=== PROBE GATE ===", flush=True)
    print(f"n_R0 = {n_r0}  (gate: = 35)", flush=True)

    arch = json.load(open(args.archive))
    arch_fq = {p["step"]: p for p in arch["curves"]["fq"]}

    for s in (0, 10, 20, 30):
        new = curves["fq"][s // 10]
        if s in arch_fq:
            old = arch_fq[s]
            d_tr = new["train_fit"] - old["train_fit"]
            d_ho = new["heldout_fit"] - old["heldout_fit"]
            print(f"step {s}: train {new['train_fit']:.6f} (old {old['train_fit']:.6f}, diff {d_tr:+.2e})  "
                  f"heldout {new['heldout_fit']:.6f} (old {old['heldout_fit']:.6f}, diff {d_ho:+.2e})", flush=True)

    # 直接对应结论的量：step 30 的 (train_fit - heldout_fit) 两条路径各是多少、差多少
    if 30 in arch_fq:
        old = arch_fq[30]
        new = curves["fq"][3]
        gap_old = old["train_fit"] - old["heldout_fit"]
        gap_new = new["train_fit"] - new["heldout_fit"]
        print(f"\nstep 30 GAP (train-heldout): old={gap_old:.6f}  new={gap_new:.6f}  diff={gap_new - gap_old:+.2e}  "
              f"(criterion: |diff| << 0.0003)", flush=True)

    for name in ("fq", "vcpo"):
        ts = step_times[name]
        if ts:
            print(f"{name}: {len(ts)} steps, mean {np.mean(ts):.2f}s/step, median {np.median(ts):.2f}s, "
                  f"min {min(ts):.2f}s, max {max(ts):.2f}s", flush=True)
    print(f"peak memory allocated = {peak_gb:.2f} GB  (old path: ~52 GB)", flush=True)


if __name__ == "__main__":
    main()
