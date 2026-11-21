#!/usr/bin/env python3
"""任务 1：复现闸门。重算 0.8B/humaneval 的 cos_Fq / cos_VCPO（hsd/opid/edge_opd）
和 grad_Fq / grad_VCPO 的 L2 范数，与 RESULTS_AUDIT.json 参照值逐格比对。

环境：dtype bf16, seed 0, n_tasks 64, n_reward_1 32。只报相对误差，不写结果文件。
"""
import os
import sys
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import cosine, load_model
from e6_alignment import build_batch, grad_F_q, grad_vcpo, grad_hsd, grad_opid, grad_edge_opd

REF = {
    "edge_opd": {"Fq": 0.35049431, "VCPO": 0.13735122},
    "hsd":      {"Fq": 0.20774829, "VCPO": 0.06825761},
    "opid":     {"Fq": 0.32295168, "VCPO": 0.04450710},
}
REF_L2 = {"fq": 340.0, "vcpo": 688.0}


def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    n_tasks = 64
    common.set_seed(0)
    model, tokenizer = load_model(model_id, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    batch = build_batch(model, tokenizer, n_tasks, device, shuffle_seed=0)
    print(f"batch: n={len(batch)}  n_reward_1={sum(1 for b in batch if b['reward']>0)}  "
          f"n_reward_0={sum(1 for b in batch if b['reward']==0)}", flush=True)

    g_fq = grad_F_q(model, tokenizer, batch, device).cpu()
    g_vcpo = grad_vcpo(model, tokenizer, batch, device).cpu()

    methods = {
        "hsd": grad_hsd(model, tokenizer, batch, device).cpu(),
        "opid": grad_opid(model, tokenizer, batch, device).cpu(),
    }
    g_edge, keep = grad_edge_opd(model, tokenizer, batch, device)
    methods["edge_opd"] = g_edge.cpu()

    print(f"L2: fq={g_fq.norm().item():.1f} (ref {REF_L2['fq']})  "
          f"vcpo={g_vcpo.norm().item():.1f} (ref {REF_L2['vcpo']})", flush=True)
    print(f"edge mask keep rate = {keep:.4f}", flush=True)

    print(f"\n{'component':<10}{'col':<8}{'recomputed':>14}{'saved':>14}{'rel_err':>12}", flush=True)
    ok = True
    for name, g in methods.items():
        for col, ref_g in (("Fq", g_fq), ("VCPO", g_vcpo)):
            v = cosine(g, ref_g)
            saved = REF[name][col]
            rel = abs(v - saved) / abs(saved) if saved else float("nan")
            print(f"{name:<10}{col:<8}{v:>14.8f}{saved:>14.8f}{rel:>12.2e}", flush=True)
            if rel > 0.02:
                ok = False
    print(f"\n{'PASS' if ok else 'FAIL'} (rel_err < 2%)", flush=True)


if __name__ == "__main__":
    main()
