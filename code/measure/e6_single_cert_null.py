#!/usr/bin/env python3
"""任务 2：单条证书零分布（0.8B / 2B, humaneval）。
ESS≈1 -> 置换权重 = 随便挑一条证书。所以穷举 32 条证书各自的余弦 = 完整零分布。
每条证书一次反向，点乘完就释放；任何时刻只持 3 个诊断梯度 + 1 条证书梯度。
自检：cos_single_at_i_star 应 ≈ 存档 cos_with_grad_VCPO（差 >5% 停）。
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
from e6_alignment import (build_batch, grad_hsd, grad_opid, grad_edge_opd,
                          grad_of, token_logprobs)

SAVED = {
    "Qwen/Qwen3.5-0.8B": {"hsd": 0.06825761, "opid": 0.04450710, "edge_opd": 0.13735122},
    "Qwen/Qwen3.5-2B": {"hsd": 0.11611987, "opid": 0.08152987, "edge_opd": 0.15629818},
}


def flat_dot(a, b):
    return float((a.float() * b.float()).sum().item())


def flat_norm(a):
    return float(a.float().norm().item())


def grad_single(model, tokenizer, prompt, cert, device):
    ids, start = common.encode_with_boundary(tokenizer, prompt, cert)
    log_p = token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -log_p).cpu()  # = ∇(-log p)，与 grad_vcpo 的 -total 同号


def main():
    model_id = sys.argv[1]
    n_tasks = 64
    common.set_seed(0)
    model, tokenizer = load_model(model_id, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    batch = build_batch(model, tokenizer, n_tasks, device, dataset="humaneval", shuffle_seed=0)
    r1 = [b for b in batch if b["reward"] > 0]  # 32 条 R=1 证书

    # 权重 + i_star（权重 1.0 那条）
    log_cs = []
    for item in r1:
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(len(batch)) + log_p, 0.0))
        log_cs.append(log_p - log_D)
    max_log_c = max(log_cs)
    a = np.array([np.exp(lc - max_log_c) for lc in log_cs])
    i_star = int(np.argmax(a))
    ess = float((a.sum() ** 2) / (a ** 2).sum())

    # 3 个诊断梯度（常驻 CPU）
    diag = {}
    g = grad_hsd(model, tokenizer, batch, device).cpu(); diag["hsd"] = g
    g = grad_opid(model, tokenizer, batch, device).cpu(); diag["opid"] = g
    g, _ = grad_edge_opd(model, tokenizer, batch, device); diag["edge_opd"] = g.cpu()

    norm_d = {k: flat_norm(v) for k, v in diag.items()}
    ip = {k: [] for k in diag}
    norm_i = []
    for item in r1:
        gi = grad_single(model, tokenizer, item["prompt"], item["certificate"], device)
        norm_i.append(flat_norm(gi))
        for k, gd in diag.items():
            ip[k].append(flat_dot(gd, gi))
        del gi
        torch.cuda.empty_cache()

    per_component = {}
    selfcheck_fail = False
    for k in diag:
        cos_all = [ip[k][i] / (norm_d[k] * norm_i[i]) for i in range(len(r1))]
        at_star = cos_all[i_star]
        saved = SAVED[model_id][k]
        rel = abs(at_star - saved) / abs(saved)
        # rank: 1 = 最高；percentile: at_star 在 32 个值里的百分位（< 越靠前越"低"）
        rank = int(np.sum(np.array(cos_all) >= at_star))
        pct = float(np.mean(np.array(cos_all) <= at_star)) * 100
        arr = np.array(cos_all)
        if pct > 70:
            verdict = "above"
        elif pct >= 30:
            verdict = "middle"
        elif pct >= 10:
            verdict = "boundary"
        else:
            verdict = "below"
        per_component[k] = {
            "cos_with_grad_VCPO_saved": saved,
            "cos_single_all": [round(float(x), 6) for x in cos_all],
            "cos_single_at_i_star": round(float(at_star), 6),
            "null_mean": round(float(arr.mean()), 4), "null_sd": round(float(arr.std()), 4),
            "null_min": round(float(arr.min()), 4), "null_max": round(float(arr.max()), 4),
            "null_q25": round(float(np.percentile(arr, 25)), 4),
            "null_median": round(float(np.percentile(arr, 50)), 4),
            "null_q75": round(float(np.percentile(arr, 75)), 4),
            "rank_of_i_star": rank, "percentile_of_i_star": round(pct, 2),
            "verdict": verdict, "selfcheck_rel_err": round(rel, 4),
        }
        print(f"[{model_id.split('/')[-1]}] {k}: at_i_star={at_star:.4f} saved={saved:.4f} "
              f"rel={rel*100:.2f}%  rank={rank}/32  pct={pct:.1f}  verdict={verdict}", flush=True)
        if rel > 0.05:
            selfcheck_fail = True

    result = {
        "model": model_id.split("/")[-1], "dataset": "humaneval", "N_certificates": len(r1),
        "ess": round(ess, 4), "i_star": i_star,
        "weights_descending": [round(float(x), 4) for x in np.sort(a)[::-1]],
        "per_component": per_component,
    }
    with open(f"data/results/e6_single_cert_null_{model_id.split('/')[-1]}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE e6_single_cert_null_{model_id.split('/')[-1]}.json  selfcheck_fail={selfcheck_fail}", flush=True)


if __name__ == "__main__":
    main()
