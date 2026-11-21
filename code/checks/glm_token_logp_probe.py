#!/usr/bin/env python3
"""GLM/HumanEval 梯度爆炸定位：逐 token 对数概率分布（前向，无反传）。

对每个 model × {clean, mismatched} 测 teacher-forced 逐 token log-prob：
- clean：R=1 证书（每题配自己的 rollout = 证书）
- mismatched：与 e6_alignment.grad_F_q_shuffled 完全相同的 deranged 配对（seed=0），
  作用在 mixed rollout 文本上、保持行标签。

错配逻辑逐字照抄 grad_F_q_shuffled 内那一段，不重新实现。
用法：python glm_token_logp_probe.py <model_id> [<model_id> ...]
结果 append 到 data/results/glm_token_logp_probe.json（12 条）。
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
from e6_alignment import build_batch, token_logprobs

OUT = "data/results/glm_token_logp_probe.json"
N_TASKS = 64


def deranged_pairing(batch, seed=0):
    """与 grad_F_q_shuffled 完全相同的错配：固定 seed 的 deranged 配对，作用在 mixed rollout 上、保持行标签。"""
    rng = random.Random(seed)
    rollouts = [item["rollout"] for item in batch]
    for _ in range(100):
        rng.shuffle(rollouts)
        if all(r != item["rollout"] for r, item in zip(rollouts, batch)):
            break
    return [dict(item, rollout=r) for item, r in zip(batch, rollouts)]


def collect_logps(model, tokenizer, device, pairs):
    all_lp = []
    for prompt, text in pairs:
        ids, start = common.encode_with_boundary(tokenizer, prompt, text)
        with torch.no_grad():
            lp = token_logprobs(model, ids.to(device), start)
        all_lp.extend(float(x) for x in lp.cpu().tolist())
    return np.array(all_lp, dtype=np.float64)


def summarize(logps):
    n = len(logps)
    if n == 0:
        return {"n_tokens_total": 0}
    sorted_lp = np.sort(logps)  # ascending = most negative first
    sum_all = float(logps.sum())
    worst10 = sorted_lp[:10]
    worst1pct = sorted_lp[:max(1, int(0.01 * n))]
    pct = lambda p: float(np.percentile(logps, p))
    return {
        "n_tokens_total": n,
        "logp_mean": round(float(logps.mean()), 4),
        "logp_median": round(float(np.median(logps)), 4),
        "logp_percentiles": {
            "p1": round(pct(1), 4), "p5": round(pct(5), 4), "p25": round(pct(25), 4),
            "p50": round(pct(50), 4), "p75": round(pct(75), 4), "p95": round(pct(95), 4),
            "p99": round(pct(99), 4),
        },
        "logp_min": round(float(logps.min()), 4),
        "n_below_minus10": int(np.sum(logps < -10)),
        "n_below_minus20": int(np.sum(logps < -20)),
        "n_below_minus30": int(np.sum(logps < -30)),
        "frac_below_minus20": round(float(np.sum(logps < -20)) / n, 6),
        "tail_mass": {
            "sum_logp_all": round(sum_all, 4),
            "sum_logp_worst_10_tokens": round(float(worst10.sum()), 4),
            "frac_of_total_from_worst_10": round(float(worst10.sum() / sum_all), 6),
            "frac_of_total_from_worst_1pct": round(float(worst1pct.sum() / sum_all), 6),
        },
    }


def probe_one(model_id):
    common.set_seed(0)
    model, tokenizer = load_model(model_id, train=False)
    device = next(model.parameters()).device
    batch = build_batch(model, tokenizer, N_TASKS, device, dataset="humaneval", shuffle_seed=0)
    shuffled = deranged_pairing(batch, seed=0)

    r1_idx = [i for i, item in enumerate(batch) if item["reward"] > 0]
    clean_pairs = [(batch[i]["prompt"], batch[i]["rollout"]) for i in r1_idx]
    mism_pairs = [(batch[i]["prompt"], shuffled[i]["rollout"]) for i in r1_idx]

    clean_logps = collect_logps(model, tokenizer, device, clean_pairs)
    mism_logps = collect_logps(model, tokenizer, device, mism_pairs)

    clean_rec = {"model": model_id.split("/")[-1], "text_condition": "clean"}
    clean_rec.update(summarize(clean_logps))
    mism_rec = {"model": model_id.split("/")[-1], "text_condition": "mismatched"}
    mism_rec.update(summarize(mism_logps))

    print(json.dumps(clean_rec), flush=True)
    print(json.dumps(mism_rec), flush=True)

    del model
    torch.cuda.empty_cache()
    return clean_rec, mism_rec


def main():
    model_ids = sys.argv[1:]
    try:
        with open(OUT) as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        records = []
    for mid in model_ids:
        try:
            c, m = probe_one(mid)
            records.append(c)
            records.append(m)
        except Exception as e:
            print(json.dumps({"model": mid.split("/")[-1], "error": str(e)[:200]}), flush=True)
            records.append({"model": mid.split("/")[-1], "error": str(e)[:200]})
        with open(OUT, "w") as f:
            json.dump(records, f, indent=2)
    print(f"WROTE {OUT} ({len(records)} records)", flush=True)


if __name__ == "__main__":
    main()
