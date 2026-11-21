#!/usr/bin/env python3
"""E2(b) 快路径探针：关 gradient checkpointing + 先切 hidden state 再过 lm_head。

评估路径（``batch_token_logprobs`` / ``mean_logprob``）逐位不变；只改训练路径
``_chunked_backward``：transformer body 只算一次 hidden state，再只把 completion
段过 lm_head，避免把整条 [Lmax, vocab] 的 prompt 部分也投影出来。

探针跑 ``--steps 30``，与存档 ``e2b_Qwen3.5-9B_seed0.json`` 的 curves.fq 逐点对，
只报数、不判定。
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
    CHUNK, _sgd_step, batch_token_logprobs, build_train_with_rewards,
    encode_batch, mean_logprob,
)


def _body_lm_head(model):
    """Return (transformer body, lm_head) for the supported architectures."""
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        return model.model, model.lm_head  # Qwen3.5 / OLMo
    if hasattr(model, "transformer") and hasattr(model, "output_layer"):
        return model.transformer, model.output_layer  # GLM
    raise RuntimeError("unknown LM structure: no model/lm_head or transformer/output_layer")


def _chunked_backward_fast(model, ids, starts, lens, mask, weights, device, normalize, lr=5e-4, chunk=CHUNK):
    """训练路径快版：body 前向一次 → 只把 completion 段过 lm_head。"""
    n = len(starts)
    model.zero_grad(set_to_none=True)
    body, lm_head = _body_lm_head(model)
    for b0 in range(0, n, chunk):
        b1 = min(b0 + chunk, n)
        ids_c = ids[b0:b1].to(device)
        mask_c = mask[b0:b1].to(device)
        # 只做 body（注意力 + FFN），不投影到 151936 词表
        hidden = body(ids_c, attention_mask=mask_c).last_hidden_state  # [c, Lmax, H] bf16

        slices = []  # (weight, hidden_slice [clen,H], target [clen])
        for j in range(b1 - b0):
            b = b0 + j
            w = weights[b]
            if w == 0.0:
                continue
            s = starts[b]
            L = lens[b]
            h = hidden[j, s - 1 : L - 1]
            t = ids_c[j, s:L]
            assert h.shape[0] == t.shape[0], "hidden slice length != target length"
            slices.append((w, h, t))

        if not slices:
            del hidden
            continue

        # 记录每条 offset（cat 后切回用同一组 offset，防静默错位）
        offsets = []
        total = 0
        for (_, h, _) in slices:
            offsets.append(total)
            total += int(h.shape[0])

        all_h = torch.cat([h for (_, h, _) in slices], dim=0)  # [total_clen, H]
        assert all_h.shape[0] == total, f"cat len {all_h.shape[0]} != sum {total}"
        logits = lm_head(all_h).float()                         # [total_clen, vocab] float32
        log_probs = torch.log_softmax(logits, dim=-1)

        chunk_total = None
        for k, (w, h, t) in enumerate(slices):
            clen = t.shape[0]
            assert clen == int(h.shape[0]), "segment length mismatch on re-slice"
            lp = log_probs[offsets[k]:offsets[k] + clen].gather(1, t.unsqueeze(1)).squeeze(1).sum()
            contrib = w * lp
            chunk_total = contrib if chunk_total is None else chunk_total + contrib
        assert offsets[-1] + int(slices[-1][1].shape[0]) == all_h.shape[0], "offset walk != cat total"

        if chunk_total is not None:
            loss = -chunk_total / n if normalize else -chunk_total
            loss.backward()
            del loss
        del hidden, all_h, logits, log_probs
    _sgd_step(model, lr)


def train_step_fq_fast(model, ids, starts, lens, mask, rewards, device, lr=5e-4):
    _chunked_backward_fast(model, ids, starts, lens, mask, rewards, device, normalize=True, lr=lr)


def train_step_vcpo_fast(model, ids, starts, lens, mask, rewards, device, lr=5e-4):
    with torch.no_grad():
        detached = batch_token_logprobs(model, ids, starts, lens, mask, device)
        log_ps = [float(pt.sum()) for pt in detached]
    n = len(starts)
    log_cs = []
    for r, log_p in zip(rewards, log_ps):
        if r > 0.0:
            log_D = float(np.logaddexp(np.log(n) + log_p, 0.0))
            log_cs.append(log_p - log_D)
        else:
            log_cs.append(None)
    max_log_c = max(lc for lc in log_cs if lc is not None)
    weights = [0.0 if lc is None else float(np.exp(lc - max_log_c)) for lc in log_cs]
    _chunked_backward_fast(model, ids, starts, lens, mask, weights, device, normalize=False, lr=lr)


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
    # 采样阶段 eval 模式（与 e2b_overfit 一致，train=True 会破坏 generate）
    model, tokenizer = load_model(args.model, train=False)
    device = next(model.parameters()).device
    # 结构自检（fail fast，别在慢采样之后才崩）
    try:
        _body, _head = _body_lm_head(model)
        print(f"structure OK: body={type(_body).__name__} lm_head={type(_head).__name__}", flush=True)
    except RuntimeError as e:
        print(f"STRUCTURE FAIL: {e}", flush=True)
        raise

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

    # 训练模型：关 checkpointing（快路径）
    model, _ = load_model(args.model, train=True)
    model.gradient_checkpointing_disable()
    model.requires_grad_(True)
    device = next(model.parameters()).device
    print(f"gradient_checkpointing = {model.is_gradient_checkpointing}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    curves = {}
    step_times = {}
    for name, step_fn in (("fq", train_step_fq_fast), ("vcpo", train_step_vcpo_fast)):
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
    print(f"n_R0 = {n_r0}  (gate: must be 35)", flush=True)

    # load archive
    try:
        arch = json.load(open(args.archive))
        arch_fq = {p["step"]: p for p in arch["curves"]["fq"]}
    except Exception as e:
        print(f"ARCHIVE READ FAILED: {e}", flush=True)
        arch_fq = {}

    for s in (0, 10, 20, 30):
        new = curves["fq"][s // 10]
        if s in arch_fq:
            old = arch_fq[s]
            d_tr = new["train_fit"] - old["train_fit"]
            d_ho = new["heldout_fit"] - old["heldout_fit"]
            print(f"step {s}: train {new['train_fit']:.6f} (old {old['train_fit']:.6f}, diff {d_tr:+.2e})  "
                  f"heldout {new['heldout_fit']:.6f} (old {old['heldout_fit']:.6f}, diff {d_ho:+.2e})", flush=True)
        else:
            print(f"step {s}: train {new['train_fit']:.6f} heldout {new['heldout_fit']:.6f}  (no archive)", flush=True)

    # 直接对应结论的量：step 30 的 (train_fit - heldout_fit) 两条路径各是多少、差多少。
    # 判据：|diff| 远小于最紧的种子噪声 sd=0.0003（否则 300 步的轨迹分岔一定淹没效应）。
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
            print(f"{name}: {len(ts)} steps, mean {np.mean(ts):.2f}s/step, "
                  f"median {np.median(ts):.2f}s, min {min(ts):.2f}s, max {max(ts):.2f}s", flush=True)
    print(f"peak memory allocated = {peak_gb:.2f} GB  (old path: ~52 GB)", flush=True)


if __name__ == "__main__":
    main()
