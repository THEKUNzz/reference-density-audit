#!/usr/bin/env python3
"""任务 0：算 E6 批次（humaneval, n_tasks=64, 32 条证书）上 VCPO 权重的 ESS。

不跑任何反向，只做 32 条证书的 teacher-forcing log-prob 前向，
复现 grad_vcpo 里的权重 a_i = exp(log_c - max log_c)，再算 ESS = (Σa)²/Σa²。

用法: python e6_ess_probe.py <model_id>
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model
from e6_alignment import token_logprobs


def main():
    model_id = sys.argv[1]
    n_tasks = 64  # humaneval batch: 32 R=1 (even indices) + 32 R=0

    model, tokenizer = load_model(model_id, train=False)
    device = next(model.parameters()).device

    import random
    items = common.load_dataset_items("humaneval")
    items = random.Random(0).sample(items, min(n_tasks, len(items)))  # shuffle_seed=0

    # R=1 项 = 偶数下标 0,2,...,62（与 e6_alignment.build_batch 一致）
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
    ess = float((a.sum() ** 2) / (a ** 2).sum())
    N = int(len(a))

    print(f"model={model_id}")
    print(f"N_certificates={N}  ESS={ess:.4f}")
    print(f"weights desc (top 10): {np.sort(a)[::-1][:10].round(4).tolist()}")
    print(f"weight sum={a.sum():.4f}  sum_sq={ (a**2).sum():.4f}")


if __name__ == "__main__":
    main()
