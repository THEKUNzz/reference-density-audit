#!/usr/bin/env python3
"""两件事：
1) 2B top-2 验证：用 2B 权重最大的前两条证书、按实际权重求加权和 grad_top2，
   和存档 cos_with_grad_VCPO 比对。贴回 <5% 就排除「单条近似」以外的 bug。
2) MBPP 的 log_c 间距：max(log_c) - 第二大的 log_c，解释为什么第二权重精确为 0。
"""
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model, cosine
from e6_alignment import build_batch, grad_hsd, grad_opid, grad_edge_opd, grad_of, token_logprobs

SAVED_2B = {"hsd": 0.11611987, "opid": 0.08152987, "edge_opd": 0.15629818}
MODELS_MBPP = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
               "allenai/OLMo-2-1124-7B", "THUDM/glm-4-9b"]


def log_c_values(model, tokenizer, device, items, n_tasks):
    out = []
    for index in range(0, n_tasks, 2):
        item = items[index]
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(n_tasks) + log_p, 0.0))
        out.append(log_p - log_D)
    return out


def main():
    # ---- Part 1: 2B top-2 验证 ----
    model_id = "Qwen/Qwen3.5-2B"
    common.set_seed(0)
    model, tokenizer = load_model(model_id, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device
    batch = build_batch(model, tokenizer, 64, device, dataset="humaneval", shuffle_seed=0)
    r1 = [b for b in batch if b["reward"] > 0]

    # log_p + log_c
    log_ps = []
    for item in r1:
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_ps.append(float(token_logprobs(model, ids.to(device), start).sum()))
    log_Ds = [float(np.logaddexp(np.log(64) + lp, 0.0)) for lp in log_ps]
    log_cs = [lp - lD for lp, lD in zip(log_ps, log_Ds)]
    max_log_c = max(log_cs)
    a = np.array([np.exp(lc - max_log_c) for lc in log_cs])
    order = np.argsort(a)[::-1]
    i1, i2 = int(order[0]), int(order[1])
    print(f"2B top2 weights: a[{i1}]={a[i1]:.4f} a[{i2}]={a[i2]:.4f}  ESS={ (a.sum()**2)/(a**2).sum():.3f}", flush=True)

    # 诊断梯度（3 个，常驻）
    g_hsd = grad_hsd(model, tokenizer, batch, device).cpu()
    g_opid = grad_opid(model, tokenizer, batch, device).cpu()
    g_edge, _ = grad_edge_opd(model, tokenizer, batch, device)
    g_edge = g_edge.cpu()

    # top-2 加权和（实际权重）
    def logp_of(item):
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        return token_logprobs(model, ids.to(device), start).sum()
    total = a[i1] * logp_of(r1[i1]) + a[i2] * logp_of(r1[i2])
    g_top2 = grad_of(model, -total).cpu()

    print("\n2B top-2 验证（cos(grad_d, grad_top2) vs 存档 cos_with_grad_VCPO）：", flush=True)
    for d_name, g_d in (("hsd", g_hsd), ("opid", g_opid), ("edge_opd", g_edge)):
        cos_v = cosine(g_d, g_top2)
        saved = SAVED_2B[d_name]
        rel = abs(cos_v - saved) / abs(saved)
        print(f"  {d_name}: top2={cos_v:.5f}  saved={saved:.5f}  rel={rel*100:.2f}%", flush=True)

    # ---- Part 2: MBPP log_c 间距 ----
    print("\nMBPP log_c 间距（max - 第二大，nats）：", flush=True)
    for mid in MODELS_MBPP:
        try:
            m2, tok2 = load_model(mid, train=False)
        except Exception as e:
            print(f"  {mid}: load fail {str(e)[:60]}", flush=True)
            continue
        dev2 = next(m2.parameters()).device
        items = common.load_dataset_items("mbpp")[:16]
        lcs = log_c_values(m2, tok2, dev2, items, 16)
        lcs_sorted = sorted(lcs, reverse=True)
        gap = lcs_sorted[0] - lcs_sorted[1]
        print(f"  {mid.split('/')[-1]:20s} max={lcs_sorted[0]:.2f} 2nd={lcs_sorted[1]:.2f} gap={gap:.2f} nats", flush=True)
        del m2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
