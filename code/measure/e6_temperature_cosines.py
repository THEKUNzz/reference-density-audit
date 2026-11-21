#!/usr/bin/env python3
"""任务 4b：温度族余弦（仅 Qwen3.5-0.8B / humaneval，要梯度）。

对 T ∈ {1, T_half, T_90, 64}，用 a_i(T) = exp((log_c_i - max_log_c)/T) 做加权累加
得到 g_w(T)，算三个诊断组件和它的余弦 + ||g_w(T)||，以及 cos(g_d, grad_F_q) 作锚点。

加权累加逐字复刻 canonical grad_vcpo（ebb2d16b 的 inline 实现），只把
coeff = exp(log_c - max_log_c) 换成 coeff = exp((log_c - max_log_c) / T)。

自检：T=1 的余弦必须贴回存档值 0.06826 / 0.04451 / 0.13735 到 5% 以内，否则停。
T_half / T_90 从 e6_temperature_sweep.json 读（0.8B/humaneval 格）。
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model, cosine
from e6_alignment import (build_batch, grad_hsd, grad_opid, grad_edge_opd,
                          grad_F_q, grad_of, token_logprobs)

SAVED = {"hsd": 0.06825761, "opid": 0.04450710, "edge_opd": 0.13735122}
MODEL_ID = "Qwen/Qwen3.5-0.8B"
N_TASKS = 64
SWEEP_PATH = "data/results/e6_temperature_sweep.json"


def load_T_half_T_90():
    with open(SWEEP_PATH) as f:
        rows = json.load(f)
    for r in rows:
        if r.get("model") == "Qwen3.5-0.8B" and r.get("dataset") == "humaneval":
            return r["T_half"], r["T_90"]
    return None, None


def grad_weighted_T(model, tokenizer, batch, device, log_cs, max_log_c, T):
    """复刻 canonical grad_vcpo 的加权累加，带温度。"""
    total = 0.0
    for item, log_c in zip(batch, log_cs):
        if log_c == -np.inf:
            continue
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        coeff = float(np.exp((log_c - max_log_c) / T))
        total = total + coeff * token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -total)


def main():
    T_half, T_90 = load_T_half_T_90()
    if T_half is None or T_90 is None:
        print("FATAL: sweep JSON 缺 0.8B/humaneval 的 T_half/T_90", flush=True)
        sys.exit(1)

    common.set_seed(0)
    model, tokenizer = load_model(MODEL_ID, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    batch = build_batch(model, tokenizer, N_TASKS, device, dataset="humaneval", shuffle_seed=0)

    # log_c 向量（batch 对齐，R=0 → -inf），与 canonical grad_vcpo 完全一致
    log_cs = []
    for item in batch:
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(len(batch)) + log_p, 0.0))
        log_c = log_p + (0.0 if item["reward"] > 0 else -np.inf) - log_D
        log_cs.append(log_c)
    max_log_c = max(log_cs)
    N = sum(1 for item in batch if item["reward"] > 0)

    # 锚点 + 三个诊断梯度（常驻 CPU）
    g_fq = grad_F_q(model, tokenizer, batch, device).cpu()
    g_hsd = grad_hsd(model, tokenizer, batch, device).cpu()
    g_opid = grad_opid(model, tokenizer, batch, device).cpu()
    g_edge, _ = grad_edge_opd(model, tokenizer, batch, device)
    g_edge = g_edge.cpu()
    diag = {"hsd": g_hsd, "opid": g_opid, "edge_opd": g_edge}

    cos_with_grad_Fq = {k: round(cosine(v, g_fq), 5) for k, v in diag.items()}

    by_T = {}
    selfcheck_fail = False
    for T in (1.0, T_half, T_90, 64.0):
        g_w = grad_weighted_T(model, tokenizer, batch, device, log_cs, max_log_c, T).cpu()
        norm = float(g_w.norm().item())
        cos_row = {k: round(cosine(v, g_w), 5) for k, v in diag.items()}
        # ESS at this T（对 R=1 证书的权重）
        w = np.array([np.exp((lc - max_log_c) / T) for lc in log_cs if lc != -np.inf])
        ess = float((w.sum() ** 2) / (w ** 2).sum())
        key = "1" if abs(T - 1.0) < 1e-9 else f"{T:g}"
        by_T[key] = {"ess": round(ess, 4), "grad_norm": round(norm, 2), "cos": cos_row}

        if abs(T - 1.0) < 1e-9:
            print(f"[self-check T=1] cos={cos_row}", flush=True)
            for k, saved in SAVED.items():
                rel = abs(cos_row[k] - saved) / abs(saved)
                print(f"  {k}: got={cos_row[k]:.5f} saved={saved:.5f} rel={rel*100:.2f}%", flush=True)
                if rel > 0.05:
                    selfcheck_fail = True
        del g_w
        torch.cuda.empty_cache()

    result = {
        "model": "Qwen3.5-0.8B", "dataset": "humaneval", "N": N,
        "T_half": T_half, "T_90": T_90,
        "cos_with_grad_Fq": cos_with_grad_Fq,
        "by_T": by_T,
        "selfcheck_pass": not selfcheck_fail,
    }
    with open("data/results/e6_temperature_cosines.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE e6_temperature_cosines.json  selfcheck_fail={selfcheck_fail}", flush=True)
    if selfcheck_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
