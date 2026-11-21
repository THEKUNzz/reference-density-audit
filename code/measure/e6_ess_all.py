#!/usr/bin/env python3
"""任务 3：18 格（6 模型 × 3 数据集）ESS 全量，无反向传播。
批次构造与 E6 完全一致：humaneval=shuffle_seed 0 / n_tasks 64；mbpp/quixbugs=前缀 / n_tasks 16。
只做 teacher-forcing log-prob 前向，套 grad_vcpo 的权重公式算 ESS。
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


def ess_for(model, tokenizer, device, items, n_tasks):
    log_cs = []
    for index in range(0, n_tasks, 2):  # R=1 = 偶数下标
        item = items[index]
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(n_tasks) + log_p, 0.0))
        log_cs.append(log_p - log_D)
    max_log_c = max(log_cs)
    return np.array([np.exp(lc - max_log_c) for lc in log_cs])


def main():
    out = []
    for model_id in MODELS:
        try:
            model, tokenizer = load_model(model_id, train=False)
        except Exception as e:
            print(json.dumps({"model": model_id, "error": f"load: {str(e)[:120]}"}), flush=True)
            out.append({"model": model_id, "error": str(e)[:120]})
            continue
        device = next(model.parameters()).device
        for ds, (n_tasks, shuffle_seed) in DATASETS.items():
            try:
                items = common.load_dataset_items(ds)
                if shuffle_seed is not None:
                    items = random.Random(shuffle_seed).sample(items, min(n_tasks, len(items)))
                else:
                    items = items[:n_tasks]
                a = ess_for(model, tokenizer, device, items, n_tasks)
                N = len(a)
                ess = float((a.sum() ** 2) / (a ** 2).sum())
                row = {
                    "model": model_id.split("/")[-1], "dataset": ds, "N_certificates": N,
                    "ess": round(ess, 4), "ess_over_N": round(ess / N, 4),
                    "weight_sum": round(float(a.sum()), 4), "weight_sumsq": round(float((a ** 2).sum()), 4),
                    "weight_max": round(float(a.max()), 6),
                    "top5_weights": [round(float(x), 4) for x in np.sort(a)[::-1][:5]],
                    "mechanical_ceiling_sqrt_N_over_ess": round((N / ess) ** 0.5, 3) if ess > 0 else None,
                }
                print(json.dumps(row), flush=True)
                out.append(row)
            except Exception as e:
                print(json.dumps({"model": model_id.split("/")[-1], "dataset": ds, "error": str(e)[:120]}), flush=True)
                out.append({"model": model_id.split("/")[-1], "dataset": ds, "error": str(e)[:120]})
        del model
        torch.cuda.empty_cache()

    with open("data/results/e6_ess_all.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE e6_ess_all.json", flush=True)


if __name__ == "__main__":
    main()
