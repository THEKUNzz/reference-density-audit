#!/usr/bin/env python3
"""E2(b) 全批次 cos 诊断（修正 DIAG_N=16 的口径问题）。

``e2b_overfit.py`` 里的 ``diagnostic_cos_fq_sft`` 只取训练批次的前 16 行做一次
不分块的 backward（这是防 OOM 的挡板，不是抽样设计）。这导致三个模型的 cos 样本
口径各不相同（前 16 行覆盖的任务数随模型强弱变化），跨模型不可比。

本脚本在**全批次**上重算 cos，用分块累积梯度（CHUNK 一致），并把"奖励加权"与
"SFT"拆成两个口径：

  - ``cos_fq_vs_uniform``  — weights 全 1（= 现有 ``diagnostic_cos_fq_sft`` 的
    "SFT" 定义：同一批数据去掉奖励加权，含 R=0 失败样本）。
  - ``cos_fq_vs_certonly`` — **自检项，恒等于 1**（二值奖励下 certonly 权重与
    rewards 逐元素相同，与 F_q 严格平行；偏离 1 = 分块累积有 bug）。

**正确性验证（硬门槛）**：把 ``flat_grad_chunked`` 限制到前 16 行重算
``cos_fq_vs_uniform_first16``，必须复现已有结果文件里的数（0.8B→0.5213、
2B→0.6830、4B→0.5450），对到小数点后 3 位。对不上 = 分块实现有 bug，停下报告。

不修改 ``e2b_overfit.py`` 的任何一行；从它 import 采样与编码即可。
"""
from __future__ import annotations

import argparse

import torch

import vcpo_common as common
from vcpo_common import RunMeta, cosine, load_model, set_seed, write_result
from e2b_overfit import CHUNK, build_train_with_rewards, encode_batch


def flat_grad_chunked(model, ids, starts, lens, mask, weights, device, chunk=CHUNK):
    """Chunked gradient accumulation, read out as a flat bf16 CPU vector.

    No SGD step: gradients are only read, never applied. Each chunk does its own
    forward+backward so the autograd graph is freed per chunk (bounds memory to
    one chunk's ``[chunk, Lmax, vocab]`` logits). The denominator is the full
    batch size ``n``, not the chunk size. ``p.grad is None`` is zero-filled so
    every gradient vector has identical length/order across calls.
    """
    n = len(starts)
    model.zero_grad(set_to_none=True)
    for b0 in range(0, n, chunk):
        b1 = min(b0 + chunk, n)
        ids_c = ids[b0:b1].to(device)
        mask_c = mask[b0:b1].to(device)
        logits = model(ids_c, attention_mask=mask_c).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        total = None
        for j in range(b1 - b0):
            b = b0 + j
            w = weights[b]
            if w == 0.0:
                continue
            lp = log_probs[j, starts[b] - 1 : lens[b] - 1]
            tgt = ids_c[j, starts[b] : lens[b]]
            c = w * lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum()
            total = c if total is None else total + c
        if total is not None:
            (-total / n).backward()          # 分母是全批 n，不是 chunk 大小
        del logits, log_probs
    # p.grad 可能为 None（例如全 R=0 的 chunk 跳过 backward），必须补零，否则
    # F_q 与 SFT 的向量长度/顺序错位。遍历 model.parameters() 的完整有序列表。
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).cpu()
        for p in model.parameters()
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    # 采样阶段必须 eval 模式（train=True 会开 gradient checkpointing 破坏 generate）。
    model, tokenizer = load_model(args.model, train=False)
    device = next(model.parameters()).device

    from datasets import load_dataset
    rows = list(load_dataset("openai/openai_humaneval", split="test"))
    train_mixed = build_train_with_rewards(
        model, tokenizer, rows, device,
        n_train=args.n_train, n_rollouts=args.n_rollouts)
    n_r1 = sum(1 for _, _, r in train_mixed if r > 0)
    n_r0 = sum(1 for _, _, r in train_mixed if r == 0)
    print(f"train batch: n_R1={n_r1} n_R0={n_r0}", flush=True)

    ids, starts, lens, mask = encode_batch(tokenizer, [it[:2] for it in train_mixed])
    rewards = [it[2] for it in train_mixed]
    n = len(rewards)

    del model
    torch.cuda.empty_cache()

    # 梯度在 train 模式模型上算（与 e2b_overfit 训练循环同构）。
    model, _ = load_model(args.model, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    def uniform_weights():
        return [1.0] * n

    def certonly_weights():
        return [1.0 if r > 0 else 0.0 for r in rewards]

    # 全批：F_q vs uniform、F_q vs cert-only。
    g_fq = flat_grad_chunked(model, ids, starts, lens, mask, rewards, device)
    g_uniform = flat_grad_chunked(model, ids, starts, lens, mask, uniform_weights(), device)
    g_certonly = flat_grad_chunked(model, ids, starts, lens, mask, certonly_weights(), device)
    cos_fq_vs_uniform_full = cosine(g_fq, g_uniform)
    cos_fq_vs_certonly_full = cosine(g_fq, g_certonly)
    # R ∈ {0,1} ⟹ certonly 权重与 rewards 逐元素相同 ⟹ 两个梯度严格平行。
    # 偏离 1 说明分块累积有 bug（累加顺序不该改变方向）。
    assert abs(cos_fq_vs_certonly_full - 1.0) < 1e-3, \
        f"certonly 应恒等于 F_q，实测 cos={cos_fq_vs_certonly_full:.6f}，分块实现有问题"
    del g_uniform, g_certonly
    torch.cuda.empty_cache()

    # 正确性验证：前 16 行，分块实现必须复现已有结果（3 位小数）。
    D = 16
    g_fq_16 = flat_grad_chunked(model, ids[:D], starts[:D], lens[:D], mask[:D],
                                rewards[:D], device)
    g_uniform_16 = flat_grad_chunked(model, ids[:D], starts[:D], lens[:D], mask[:D],
                                     [1.0] * D, device)
    cos_fq_vs_uniform_first16 = cosine(g_fq_16, g_uniform_16)
    del g_fq_16, g_uniform_16
    torch.cuda.empty_cache()

    meta = RunMeta(
        experiment="e2b_cos_full",
        model=args.model,
        dataset="humaneval",
        seed=args.seed,
        dtype="bfloat16",
        extra={
            "n_train": args.n_train, "n_rollouts": args.n_rollouts,
            "n_R1": n_r1, "n_R0": n_r0,
            "chunk": CHUNK, "first16": D,
            "note": ("cos on the FULL train batch via chunked gradient accumulation; "
                     "uniform = existing 'SFT' def (mixed batch, no reward weighting); "
                     "certonly = self-check, analytically 1 under binary reward (asserted)"),
        },
    ).fill_environment()
    payload = {
        "model": args.model, "seed": args.seed,
        "n_R1": n_r1, "n_R0": n_r0,
        "cos_fq_vs_uniform_full": cos_fq_vs_uniform_full,
        "cos_fq_vs_certonly_full": cos_fq_vs_certonly_full,
        "cos_fq_vs_uniform_first16": cos_fq_vs_uniform_first16,
    }
    path = write_result(f"e2b_cos_{args.model.split('/')[-1]}_seed{args.seed}", meta, payload)

    print(f"model={args.model} n_R1={n_r1} n_R0={n_r0}")
    print(f"  cos_fq_vs_uniform_first16 = {cos_fq_vs_uniform_first16:.4f}  (must match existing)")
    print(f"  cos_fq_vs_uniform_full    = {cos_fq_vs_uniform_full:.4f}")
    print(f"  cos_fq_vs_certonly_full   = {cos_fq_vs_certonly_full:.6f}  (self-check, want 1.0)")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
