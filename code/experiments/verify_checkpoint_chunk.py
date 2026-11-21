#!/usr/bin/env python3
"""验证两个加速杠杆的数值安全性与提速幅度（只读，不训练、不写结果）。

在 0.8B 上、16 条证书上测三件事：
  1. 关 gradient checkpointing 是否数值不变（cos(∇_cp_on, ∇_cp_off) 应 ≈ 1.0）；
  2. CHUNK 4 → 16 是否改数值（cos(∇_chunk4, ∇_chunk16) 报告出来）；
  3. 各自的单次梯度耗时与峰值显存。

不采样、不建 40 题 rollout，只编码 16 条 (prompt, certificate) 直接算梯度，
几分钟跑完。可与正在跑的训练共存（0.8B 峰值 < 15GB，GPU 有 ~49GB 空余）。
"""
from __future__ import annotations

import argparse
import time

import torch

import vcpo_common as common
from vcpo_common import cosine, load_model, set_seed
from e2b_overfit import encode_batch
from e2b_cos_full import flat_grad_chunked


def time_grad(model, ids, starts, lens, mask, weights, device, chunk, repeats=3):
    """warmup + timed gradient computation; returns (mean_sec, grad)."""
    # warmup（首调触发 CUDA kernel 编译）
    flat_grad_chunked(model, ids, starts, lens, mask, weights, device, chunk=chunk)
    torch.cuda.synchronize()
    times = []
    grad = None
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        grad = flat_grad_chunked(model, ids, starts, lens, mask, weights, device, chunk=chunk)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return sum(times) / len(times), grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-items", type=int, default=16)
    args = parser.parse_args()

    set_seed(0)
    model, tokenizer = load_model(args.model, train=True)  # checkpointing ON by default
    model.requires_grad_(True)
    device = next(model.parameters()).device

    from datasets import load_dataset
    rows = list(load_dataset("openai/openai_humaneval", split="test"))[: args.n_items]
    items = [(r["prompt"], r["canonical_solution"]) for r in rows]
    ids, starts, lens, mask = encode_batch(tokenizer, items)
    weights = [1.0] * len(items)  # uniform/SFT loss, enough for the numerics question

    print(f"model={args.model}  n_items={len(items)}  Lmax={ids.shape[1]}", flush=True)
    print(f"checkpointing ON at start: {model.is_gradient_checkpointing}", flush=True)

    # --- 1) checkpointing ON, CHUNK=4 ---
    torch.cuda.reset_peak_memory_stats()
    t_on, g_cp_on = time_grad(model, ids, starts, lens, mask, weights, device, chunk=4)
    mem_on = torch.cuda.max_memory_allocated() / 1e9
    print(f"[cp ON   chunk=4]  {t_on:.2f}s/grad   peak={mem_on:.2f}GB", flush=True)

    # --- 2) checkpointing OFF, CHUNK=4 ---
    model.gradient_checkpointing_disable()
    print(f"checkpointing disabled: now {model.is_gradient_checkpointing}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_off, g_cp_off = time_grad(model, ids, starts, lens, mask, weights, device, chunk=4)
    mem_off = torch.cuda.max_memory_allocated() / 1e9
    print(f"[cp OFF  chunk=4]  {t_off:.2f}s/grad   peak={mem_off:.2f}GB", flush=True)
    print(f"cos(cp_on, cp_off) = {cosine(g_cp_on, g_cp_off):.8f}   (want ~1.0)", flush=True)

    # --- 3) checkpointing OFF, CHUNK=16 ---
    torch.cuda.reset_peak_memory_stats()
    t_16, g_chunk16 = time_grad(model, ids, starts, lens, mask, weights, device, chunk=16)
    mem_16 = torch.cuda.max_memory_allocated() / 1e9
    print(f"[cp OFF  chunk=16] {t_16:.2f}s/grad   peak={mem_16:.2f}GB", flush=True)
    print(f"cos(chunk4, chunk16) = {cosine(g_cp_off, g_chunk16):.8f}   (deviates from 1 => numerics change)", flush=True)

    print("\nSUMMARY:")
    print(f"  checkpointing ON -> OFF: {t_on:.2f}s -> {t_off:.2f}s  ({t_on/t_off:.2f}x faster), cos={cosine(g_cp_on, g_cp_off):.6f}")
    print(f"  CHUNK 4 -> 16 (cp off): {t_off:.2f}s -> {t_16:.2f}s  ({t_off/t_16:.2f}x faster), cos={cosine(g_cp_off, g_chunk16):.6f}")


if __name__ == "__main__":
    main()
