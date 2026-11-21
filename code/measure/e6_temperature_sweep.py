#!/usr/bin/env python3
"""任务 4a：温度扫描（18 格，无梯度）。

机制：现有权重 a_i = exp(log_c_i - max_log_c)，log_c = log_p - logaddexp(log(n)+log_p, 0)
≈ log_p（因为 64p << 1）。证书概率跨几十 nats，所以 a_i 被最高那条吃光。

温度族：a_i(T) = exp((log_c_i - max_log_c) / T)。T=1 是现状，T→∞ 等权。
扫 T ∈ {1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64}，报 ESS(T)；再二分解出
T_half (ESS=N/2) 和 T_90 (ESS=0.9N)。

完全不需要梯度、不需要前向——只算 log_c 的标量，几秒钟。
批次与 E6 完全一致：humaneval=shuffle_seed 0 / n_tasks 64；mbpp/quixbugs=前缀 / n_tasks 16。
R=1 = 偶数下标。
"""
import os
import sys
import random
import json
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model
from e6_alignment import token_logprobs

MODELS = [
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
    "allenai/OLMo-2-1124-7B", "THUDM/glm-4-9b",
]
DATASETS = {"humaneval": (64, 0), "mbpp": (16, None), "quixbugs": (16, None)}
T_SWEEP = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0]


def log_c_vector(model, tokenizer, device, items, n_tasks):
    """R=1 证书（偶数下标）的 log_c 向量。"""
    log_cs = []
    for index in range(0, n_tasks, 2):
        item = items[index]
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(n_tasks) + log_p, 0.0))
        log_cs.append(log_p - log_D)
    return np.array(log_cs, dtype=np.float64)


def ess_at_T(log_cs, T):
    z = log_cs - log_cs.max()
    a = np.exp(z / T)
    return float((a.sum() ** 2) / (a ** 2).sum())


def solve_T(log_cs, target_ess):
    """二分：ESS(T) 随 T 单调上升到 N。"""
    lo, hi = 1.0, 1.0
    # 扩张上界直到 ESS > target（T→∞ 时 ESS→N > target，除非 target≥N）
    guard = 0
    while ess_at_T(log_cs, hi) < target_ess:
        hi *= 2.0
        guard += 1
        if guard > 200 or hi > 1e15:
            break
    if ess_at_T(log_cs, hi) < target_ess:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ess_at_T(log_cs, mid) < target_ess:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def main():
    out = []
    for model_id in MODELS:
        try:
            model, tokenizer = load_model(model_id, train=False)
        except Exception as e:
            print(json.dumps({"model": model_id, "error": f"load: {str(e)[:120]}"}), flush=True)
            out.append({"model": model_id.split("/")[-1], "error": str(e)[:120]})
            continue
        device = next(model.parameters()).device
        for ds, (n_tasks, shuffle_seed) in DATASETS.items():
            try:
                items = common.load_dataset_items(ds)
                if shuffle_seed is not None:
                    items = random.Random(shuffle_seed).sample(items, min(n_tasks, len(items)))
                else:
                    items = items[:n_tasks]
                log_cs = log_c_vector(model, tokenizer, device, items, n_tasks)
                N = len(log_cs)
                sorted_lc = np.sort(log_cs)[::-1]
                ess_vs_T = {f"{T:g}": round(ess_at_T(log_cs, T), 4) for T in T_SWEEP}
                T_half = solve_T(log_cs, N / 2.0)
                T_90 = solve_T(log_cs, 0.9 * N)
                row = {
                    "model": model_id.split("/")[-1], "dataset": ds, "N": N,
                    "ess_vs_T": ess_vs_T,
                    "T_half": round(T_half, 3) if T_half is not None else None,
                    "T_90": round(T_90, 3) if T_90 is not None else None,
                    "logc_spread_max_minus_min": round(float(sorted_lc[0] - sorted_lc[-1]), 3),
                    "logc_spread_max_minus_second": round(float(sorted_lc[0] - sorted_lc[1]), 3),
                    "logc_max": round(float(sorted_lc[0]), 3),
                    "logc_min": round(float(sorted_lc[-1]), 3),
                }
                print(json.dumps(row), flush=True)
                out.append(row)
            except Exception as e:
                print(json.dumps({"model": model_id.split("/")[-1], "dataset": ds,
                                  "error": str(e)[:120]}), flush=True)
                out.append({"model": model_id.split("/")[-1], "dataset": ds, "error": str(e)[:120]})
        del model
        torch.cuda.empty_cache()

    with open("data/results/e6_temperature_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE e6_temperature_sweep.json", flush=True)


if __name__ == "__main__":
    main()
