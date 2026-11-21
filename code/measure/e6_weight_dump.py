#!/usr/bin/env python3
"""把两组 VCPO 权重全量 dump 出来（前缀批次 vs shuffle_seed=0 批次），并各自算 ESS + grad_VCPO 范数。

第一组 = build_batch(shuffle_seed=None) -> items[:64] 的前缀批次
第二组 = build_batch(shuffle_seed=0)  -> random.Random(0).sample(items,64)
权重公式与 grad_vcpo 完全一致：log_c = log_p - logaddexp(log(n)+log_p, 0)，coeff = exp(log_c - max_log_c)。
"""
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model
from e6_alignment import token_logprobs, grad_of


def weights_for(model, tokenizer, device, items, n_tasks):
    log_cs = []
    for index in range(0, n_tasks, 2):
        item = items[index]
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(n_tasks) + log_p, 0.0))
        log_c = log_p - log_D
        log_cs.append(log_c)
    max_log_c = max(log_cs)
    a = np.array([np.exp(lc - max_log_c) for lc in log_cs])
    return a


def grad_vcpo_norm(model, tokenizer, device, items, n_tasks):
    """按 grad_vcpo 的代码路径（-total 反向）算加权和的 L2 范数。"""
    log_cs = []
    for index in range(0, n_tasks, 2):
        item = items[index]
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(n_tasks) + log_p, 0.0))
        log_cs.append(log_p - log_D)
    max_log_c = max(log_cs)
    total = 0.0
    for index in range(0, n_tasks, 2):
        item = items[index]
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        coeff = float(np.exp(log_cs[index // 2] - max_log_c))
        total = total + coeff * token_logprobs(model, ids.to(device), start).sum()
    g = grad_of(model, -total)
    return float(g.norm().item())


def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    n_tasks = 64
    model, tokenizer = load_model(model_id, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    all_items = common.load_dataset_items("humaneval")

    # 第一组：前缀
    items1 = all_items[:n_tasks]
    w1 = weights_for(model, tokenizer, device, items1, n_tasks)
    # 第二组：shuffle_seed=0
    items2 = random.Random(0).sample(all_items, n_tasks)
    w2 = weights_for(model, tokenizer, device, items2, n_tasks)

    for name, w in (("run1_prefix", w1), ("run2_shuffle0", w2)):
        a = np.sort(w)[::-1]
        ess = (w.sum()**2) / (w**2).sum()
        print(f"\n=== {name} ===")
        print(f"sum={w.sum():.4f}  sum_sq={(w**2).sum():.4f}  ESS={ess:.4f}  max={a[0]:.6f}")
        print(f"weights(desc, 32): {[round(float(x),4) for x in a]}")

    # grad_VCPO 范数（用 grad_vcpo 的 -total 路径）
    n1 = grad_vcpo_norm(model, tokenizer, device, items1, n_tasks)
    n2 = grad_vcpo_norm(model, tokenizer, device, items2, n_tasks)
    print(f"\ngrad_VCPO L2:  run1_prefix={n1:.1f}   run2_shuffle0={n2:.1f}   (paper=688.0)")


if __name__ == "__main__":
    main()
