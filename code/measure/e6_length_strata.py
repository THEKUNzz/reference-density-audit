#!/usr/bin/env python3
"""S2 Task 1: certificate-length stratification of the EDGE-OPD vs grad_F_q cosine.

Per-task cos(grad_edge_opd, grad_F_q), split by median certificate length.
Judge: |r| > 0.3 negative => "length as confounder" holds; |r| < 0.15 => no trend.

Notes on the per-task decomposition:
- grad_F_q is reward-weighted, so it is only non-degenerate on R=1 items
  (R=0 items contribute exactly zero to F_q). We therefore restrict the
  per-task cosine to R=1 items (rollout == certificate), which is also where
  "certificate length" is a well-defined confounder.
- grad_edge_opd returns (gradient, mask_rate); we use the gradient.
"""
import os, sys, json, statistics

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
import e6_alignment as E6

MODEL = "allenai/OLMo-2-1124-7B"
N = 32
SEED = 0

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    common.set_seed(SEED)
    model, tokenizer = common.load_model(MODEL, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    batch = E6.build_batch(model, tokenizer, N, device)

    rows = []
    skipped = 0
    nan_rows = []
    for item in batch:
        if item["reward"] == 0.0:
            skipped += 1  # F_q is exactly zero on R=0 items -> cosine undefined
            continue
        single = [item]
        g_edge, mask_rate = E6.grad_edge_opd(model, tokenizer, single, device)
        g_fq = E6.grad_F_q(model, tokenizer, single, device)
        cos = common.cosine(g_edge, g_fq)
        L = len(item["certificate"])  # char length, matches earlier cert-len stats
        rec = {"task_id": item["task_id"], "cert_len": L, "cos": float(cos),
               "edge_mask_rate": float(mask_rate)}
        if cos != cos:  # NaN -> zero EDGE gradient (empty evidence mask)
            nan_rows.append(rec)
        else:
            rows.append(rec)

    lens = [r["cert_len"] for r in rows]
    coss = [r["cos"] for r in rows]
    med = statistics.median(lens)
    short = [r for r in rows if r["cert_len"] <= med]
    long_ = [r for r in rows if r["cert_len"] > med]

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    # Pearson r between cert_len and cos
    n = len(rows)
    r = float("nan")
    if n >= 2:
        mx = statistics.mean(lens)
        my = statistics.mean(coss)
        cov = sum((r_["cert_len"] - mx) * (r_["cos"] - my) for r_ in rows)
        vx = sum((r_["cert_len"] - mx) ** 2 for r_ in rows)
        vy = sum((r_["cos"] - my) ** 2 for r_ in rows)
        r = cov / ((vx * vy) ** 0.5) if vx > 0 and vy > 0 else float("nan")

    out = {
        "model": MODEL,
        "n": N,
        "n_R1": len(rows),
        "n_R0_skipped": skipped,
        "n_nan_skipped": len(nan_rows),
        "median_cert_len_chars": med,
        "short": {"n": len(short), "mean_cos": mean([x["cos"] for x in short])},
        "long": {"n": len(long_), "mean_cos": mean([x["cos"] for x in long_])},
        "pearson_r": r,
        "rows": rows,
        "nan_rows": nan_rows,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open("data/results/e6_length_strata.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("wrote data/results/e6_length_strata.json")


if __name__ == "__main__":
    main()
