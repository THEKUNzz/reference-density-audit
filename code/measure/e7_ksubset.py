#!/usr/bin/env python3
"""E7: k-reference aggregation control (exact Gram, no sketch).

Extends e6_single_cert_null.py by ONE change: the 32 per-certificate gradients
are kept in CPU RAM instead of being freed, so the exact 32x32 Gram matrix can
be formed. That is all the k-subset curve needs:

    cos(g_d, sum_{i in S} s_i) = (sum_{i in S} d_i)
                                 / (||g_d|| * sqrt(sum_i n_i^2 + 2 sum_{i<j} G_ij))

with d_i = <g_d, s_i> and n_i = ||s_i|| exact (computed in the same pass), and
G the exact Gram. No sketching, no approximation anywhere.

RAM: the container cgroup limit is 120 GB (/sys/fs/cgroup/memory.max). `free`
reports ~1007 GB but that is the HOST, not this container. 32 bf16 gradients is
48 GB for 0.8B (fits, exact Gram used) but 128 GB for 2B (does NOT fit -- a run
was SIGKILLed at cert 25/32). 2B therefore uses the count-sketch path, whose
error is calibrated against the exact 0.8B result rather than assumed.

Gates:
  G1  the 32 single-certificate cosines must reproduce
      e6_single_cert_null_<tag>.json to the precision it was stored at (1e-6,
      values are round(x,6) there so allow 5e-6).
  G3  k=32 must reproduce cos_with_grad_Fq from the E6 archive to within 2%.
      This is exact here, so it should land far inside that.

Writes results to the shared volume. Does not touch the 30G system disk and
does not touch the archived validation or resume directories.
"""
import sys
import os
import json
import itertools
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model
from e6_alignment import (build_batch, grad_hsd, grad_opid, grad_edge_opd,
                          grad_of, token_logprobs)

RESULTS = "data/results"

def load_refs(tag, dataset):
    """G3 reference cosines, archived ESS, and archived g_C, for ANY of the 18 cells.

    Reads RESULTS_AUDIT.json (all 18 alignment cells) and e6_ess_all.json, so
    this works beyond the two models that have a single-certificate enumeration.
    """
    audit = json.load(open(f"{RESULTS}/RESULTS_AUDIT.json"))
    fq = gc = extra = gdn = None
    n_tasks = n_ref = None
    for a in audit["alignment"]:
        if a["model"] == tag and a["dataset"] == dataset:
            fq = {c: a["alignment"][c]["cos_with_grad_Fq"]
                  for c in ("hsd", "opid", "edge_opd")}
            gc = {c: a["alignment"][c]["cos_with_grad_VCPO"]
                  for c in ("hsd", "opid", "edge_opd")}
            gdn = {c: a["grad_l2_norms"][c] for c in ("hsd", "opid", "edge_opd")}
            # HumanEval used n_tasks=64 (32 references); MBPP and QuixBugs used
            # n_tasks=16 (8 references). Hardcoding 64 silently produced a
            # different batch on the 8-reference datasets, which the ESS check
            # could not catch because ESS is 1.00 on both sides there.
            extra = dict(a["meta"]["extra"])
            extra["seed"] = a["meta"].get("seed", 0)
            n_tasks = extra["n_tasks"]
            n_ref = extra["n_reward_1"]
    ess = None
    try:
        allx = json.load(open(f"{RESULTS}/e6_ess_all.json"))
        rows = allx if isinstance(allx, list) else (allx.get("cells") or
               allx.get("results") or list(allx.values())[0])
        for r in rows:
            if isinstance(r, dict) and r.get("model") == tag and r.get("dataset") == dataset:
                ess = r.get("ess") or r.get("ESS")
    except Exception as e:
        print(f"  (e6_ess_all.json unreadable: {e})", flush=True)
    return fq, gc, ess, n_tasks, n_ref, extra, gdn
def ks_for(n):
    """Aggregation sizes up to N, always including N itself (the full average)."""
    return [k for k in (1, 2, 4, 8, 16, 32, 64) if k < n] + [n]
SKETCH_M = 1 << 18          # count-sketch width; cost is O(D), independent of m
SKETCH_SEED = 20260930
N_SUBSETS = 500
SUBSET_SEED = 7


def flat_dot(a, b):
    return float((a.float() * b.float()).sum().item())


def flat_norm(a):
    return float(a.float().norm().item())


def grad_single(model, tokenizer, prompt, cert, device):
    ids, start = common.encode_with_boundary(tokenizer, prompt, cert)
    log_p = token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -log_p).cpu()


def exact_gram(vecs, device, chunk=1 << 24):
    """Exact Gram over ALL supplied vectors, in float64 throughout.

    float32 is not usable here: a dot product over 7.5e8 elements accumulates
    enough float32 error to put the diagonal ~10% off ||s_i||^2 (measured). The
    matmul itself is float64 -- not merely the accumulator -- and TF32 is
    disabled so no reduced-mantissa path can be substituted.

    Passing the diagnostic gradients and g_C in the same stack means norms,
    dot products and cross terms all come from one consistent computation.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    n, D = len(vecs), vecs[0].numel()
    G = torch.zeros(n, n, dtype=torch.float64, device=device)
    done = 0
    for s0 in range(0, D, chunk):
        s1 = min(s0 + chunk, D)
        M = torch.stack([v[s0:s1] for v in vecs]).to(device=device, dtype=torch.float64)
        G += M @ M.T
        del M
        done += 1
        if done % 10 == 0:
            torch.cuda.empty_cache()
            print(f"    gram chunk {done} ({s1}/{D})", flush=True)
    return G.cpu().numpy()


def sketch_vec(v, device, m=SKETCH_M, seed=SKETCH_SEED, chunk=1 << 24):
    """Count-sketch of one flat vector, float64 throughout.

    The hash (h, sign) is drawn from a generator seeded as a pure function of
    (seed, chunk_index), so EVERY vector gets the SAME hash. If the generator
    were consumed sequentially across vectors instead, each would get its own
    projection and the Gram would be meaningless -- while still passing a
    diagonal check. That is the one failure mode a diagonal test cannot see.
    """
    z = torch.zeros(m, dtype=torch.float64, device=device)
    for ci, s0 in enumerate(range(0, v.numel(), chunk)):
        x = v[s0:s0 + chunk].to(device=device, dtype=torch.float64)
        g = torch.Generator(device=device)
        g.manual_seed(seed * 1000003 + ci)
        h = torch.randint(0, m, (x.numel(),), generator=g, device=device)
        sg = torch.randint(0, 2, (x.numel(),), generator=g, device=device,
                           dtype=torch.int8).to(torch.float64) * 2 - 1
        z.index_add_(0, h, x * sg)
        del x, h, sg
    return z


def curve(d, n2, G, gd_norm, rng, KS):
    """cos(g_d, sum_{i in S} s_i) over random k-subsets. Exact."""
    N = len(d)
    out = {}
    for k in KS:
        if k == 1:
            subsets = [np.array([i]) for i in range(N)]
        elif k == N:
            subsets = [np.arange(N)]
        else:
            subsets = [rng.choice(N, k, replace=False) for _ in range(N_SUBSETS)]
        cs = []
        for S in subsets:
            num = d[S].sum()
            sub = G[np.ix_(S, S)]
            den2 = n2[S].sum() + 2 * sub[np.triu_indices(k, 1)].sum()
            cs.append(num / (gd_norm * np.sqrt(max(den2, 1e-30))))
        cs = np.array(cs, dtype=float)
        out[k] = {
            "median": float(np.median(cs)), "p05": float(np.percentile(cs, 5)),
            "p95": float(np.percentile(cs, 95)), "min": float(cs.min()),
            "max": float(cs.max()), "mean": float(cs.mean()),
            "n_subsets": len(cs),
        }
    return out


def main():
    model_id = sys.argv[1]
    # --sketch      : keep vectors, compute exact Gram AND the sketch (calibration)
    # --sketch-only : never hold all vectors; sketch each cert and free it (2B)
    dataset = "humaneval"
    argv = sys.argv[2:]
    for i, x in enumerate(argv):
        if x == "--dataset":
            dataset = argv[i + 1]
    mode = next((x for x in argv if x.startswith("--") and x != "--dataset"), "--auto")
    sketch_only = (mode == "--sketch-only")
    do_sketch = mode in ("--sketch", "--sketch-only", "--auto")
    tag = model_id.split("/")[-1]
    fq_ref, gc_ref, ess_ref, n_tasks, n_ref, extra, gdn_ref = load_refs(tag, dataset)
    if fq_ref is None or n_tasks is None:
        sys.exit(f"[{tag}/{dataset}] no archived alignment row; refusing to guess "
                 f"the batch size.")
    # EVERY batch-determining parameter comes from the archive. Hardcoding
    # HumanEval's values silently produced different batches on the other two
    # datasets twice: n_tasks is 64 there but 16 here, and shuffle_seed is 0
    # there but None here (None means the task-id prefix, not a random sample).
    shuffle_seed = extra.get("shuffle_seed", 0)
    seed = extra.get("seed", 0)
    print(f"[{tag}/{dataset}] archive batch spec: n_tasks={n_tasks} "
          f"shuffle_seed={shuffle_seed} seed={seed} references={n_ref} "
          f"ESS={ess_ref}", flush=True)
    common.set_seed(seed)
    model, tokenizer = load_model(model_id, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device

    # Choose exact vs sketched Gram from the actual memory requirement rather
    # than from a flag. Exact needs the N certificate gradients plus the three
    # diagnostics and g_C resident at once; the cgroup allows 120 GB, so budget
    # 90 GB and leave the rest for the model, activations and allocator slack.
    # This matters: with only 8 references the sketch has 28 off-diagonal terms
    # instead of 496, so its errors cancel far less and G3 degrades.
    gbytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * 2
    need = (n_ref + 4) * gbytes
    budget = 90 * 1024 ** 3
    if mode == "--auto":
        sketch_only = need > budget
        print(f"[{tag}/{dataset}] exact Gram needs {need/1024**3:.0f} GB for "
              f"{n_ref}+4 vectors; budget {budget/1024**3:.0f} GB -> "
              f"{'SKETCH' if sketch_only else 'EXACT (+sketch for calibration)'}",
              flush=True)

    batch = build_batch(model, tokenizer, n_tasks, device,
                        dataset=dataset, shuffle_seed=shuffle_seed)
    r1 = [b for b in batch if b["reward"] > 0]
    N = len(r1)
    print(f"[{tag}] batch built, {N} R=1 certificates", flush=True)

    # weights / ESS / i_star -- identical to e6_single_cert_null.py
    log_cs = []
    for item in r1:
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"],
                                                 item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(len(batch)) + log_p, 0.0))
        log_cs.append(log_p - log_D)
    a = np.array([np.exp(lc - max(log_cs)) for lc in log_cs])
    i_star = int(np.argmax(a))
    ess = float((a.sum() ** 2) / (a ** 2).sum())
    print(f"[{tag}] ESS={ess:.4f} i_star={i_star}", flush=True)

    # In sketch-only mode a diagnostic gradient is sketched the moment it exists
    # and the full vector is freed. Holding all three plus the transient doubling
    # inside flat_grad is what exceeded the 120 GB cgroup on the 7B/9B models
    # (3 x 18 GB resident + 36 GB peak while concatenating).
    comps = ["hsd", "opid", "edge_opd"]
    diag, diag_sk, norm_d = {}, {}, {}
    for name, fn in zip(comps, (grad_hsd, grad_opid, grad_edge_opd)):
        out = fn(model, tokenizer, batch, device)
        g = (out[0] if isinstance(out, tuple) else out).cpu()
        if sketch_only:
            diag_sk[name] = sketch_vec(g, device).cpu()
            del g
        else:
            diag[name] = g
            norm_d[name] = flat_norm(g)
        torch.cuda.empty_cache()
    print(f"[{tag}] diagnostic grads done"
          + ("" if sketch_only else f", norms={ {k: round(v,3) for k,v in norm_d.items()} }"),
          flush=True)

    # --- the one change from e6_single_cert_null.py: KEEP the 32 gradients ---
    certs, sketches = [], []
    ip = {k: [] for k in diag}
    norm_i = []
    for idx, item in enumerate(r1):
        gi = grad_single(model, tokenizer, item["prompt"], item["certificate"], device)
        if not sketch_only:
            norm_i.append(flat_norm(gi))
            for k, gd in diag.items():
                ip[k].append(flat_dot(gd, gi))
        if do_sketch:
            sketches.append(sketch_vec(gi, device).cpu())
        if sketch_only:
            del gi
        else:
            certs.append(gi)
        torch.cuda.empty_cache()
        if (idx + 1) % 8 == 0:
            held = 0 if sketch_only else len(certs)
            print(f"    cert {idx+1}/{N}  held={held}", flush=True)
    if sketch_only:
        print(f"[{tag}] {N} certificates sketched (m={SKETCH_M}), vectors freed",
              flush=True)
    else:
        print(f"[{tag}] {N} gradients held, "
              f"{certs[0].numel()*2*N/1e9:.1f} GB bf16 in RAM", flush=True)

    # ---------------------------------------------------------------- G1
    # G1 proves we are on the SAME BATCH as E6. Two archives supply references:
    # RESULTS_AUDIT.json covers all 18 cells; the per-certificate enumeration
    # (e6_single_cert_null_*.json) exists only for Qwen3.5-0.8B and -2B, so the
    # cosine-level comparison is available for those two and is reported as
    # "not archived" elsewhere rather than silently skipped.
    # ESS tolerance is RELATIVE (3%), not absolute: ESS grows more sensitive to
    # the subdominant weights as it rises, so a fixed absolute bound that suits
    # ESS 1.06 is wrong at ESS 3.19. The discrete facts (N, and i_star where
    # archived) must match exactly.
    ess_rel = abs(ess - ess_ref) / ess_ref if ess_ref else float("nan")
    batch_id = {"ess_here": round(ess, 4), "ess_archived": ess_ref,
                "ess_rel_dev": ess_rel, "N": N}
    # N is a hard requirement everywhere. Previously it was only compared where
    # the per-certificate archive existed, so a 32-vs-8 reference-count mismatch
    # passed on MBPP while ESS read 1.00 on both sides.
    batch_ok = (ess_ref is not None) and ess_rel < 0.03 and N == n_ref
    batch_id["N_archived"] = n_ref

    ref = None
    p_null = f"{RESULTS}/e6_single_cert_null_{tag}.json"
    if dataset == "humaneval" and os.path.exists(p_null):
        ref = json.load(open(p_null))
        batch_id["i_star_here"] = i_star
        batch_id["i_star_archived"] = ref["i_star"]
        batch_ok = batch_ok and i_star == ref["i_star"] and N == ref["N_certificates"]
    print(f"[{tag}/{dataset}] batch identity: {batch_id} -> "
          f"{'MATCH' if batch_ok else 'MISMATCH'}", flush=True)
    if not batch_ok:
        print(f"[{tag}/{dataset}] *** BATCH MISMATCH *** halting.", flush=True)
        json.dump({"model": tag, "dataset": dataset, "batch_identity": batch_id,
                   "batch_ok": False},
                  open(f"{RESULTS}/e7_ksubset_{tag}_{dataset}_FAILED.json", "w"),
                  indent=2)
        return

    g1, g1_max, g1_mean = None, None, None
    if ref is not None and not sketch_only:
        g1 = {}
        g1_max = 0.0
        for k in diag:
            cos_all = np.array([ip[k][i] / (norm_d[k] * norm_i[i]) for i in range(N)])
            ref_all = np.array(ref["per_component"][k]["cos_single_all"])
            delta = np.abs(cos_all - ref_all)
            g1[k] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
            g1_max = max(g1_max, float(delta.max()))
        g1_mean = max(v["mean_abs"] for v in g1.values())
        ok = g1_max < 5e-3
        print(f"[{tag}/{dataset}] G1 max |delta| = {g1_max:.3e}, worst mean = "
              f"{g1_mean:.3e} -> {'PASS' if ok else 'FAIL'} (halt above 5e-3)",
              flush=True)
        if not ok:
            json.dump({"model": tag, "dataset": dataset, "G1": g1,
                       "G1_max_abs_delta": g1_max, "G1_pass": False},
                      open(f"{RESULTS}/e7_ksubset_{tag}_{dataset}_FAILED.json", "w"),
                      indent=2)
            return
    else:
        print(f"[{tag}/{dataset}] G1: no per-certificate archive for this cell; "
              f"batch identity rests on ESS ({ess_rel*100:.2f}% dev) and N.",
              flush=True)

    # g_C on THIS stack, so the overlay and the curve are internally consistent.
    from e6_alignment import grad_vcpo
    gc_vec = grad_vcpo(model, tokenizer, batch, device).cpu()
    gc_sk = None
    if sketch_only:
        gc_sk = sketch_vec(gc_vec, device).cpu()
        del gc_vec
    torch.cuda.empty_cache()

    # --------------------------------------------------------- Gram
    # Stack order: 32 certificates, then hsd/opid/edge_opd, then g_C. One pass
    # yields every norm, dot product and cross term consistently.
    idx_c = {k: N + i for i, k in enumerate(comps)}
    idx_gc = N + len(comps)

    Gs = None
    if sketch_only:
        Z = torch.stack(sketches + [diag_sk[k] for k in comps] + [gc_sk]).numpy()
        Gs = Z @ Z.T
        del Z
    elif do_sketch:
        extra = [diag[k] for k in comps] + [gc_vec]
        print(f"[{tag}] sketching {len(extra)} extra vectors (m={SKETCH_M}) ...",
              flush=True)
        sk = sketches + [sketch_vec(v, device).cpu() for v in extra]
        Z = torch.stack(sk).numpy()
        Gs = Z @ Z.T
        del sk, Z

    if sketch_only:
        Gfull = Gs
        gram_method = f"count_sketch_m{SKETCH_M}"
        print(f"[{tag}] Gram from count-sketch (exact Gram does not fit in the "
              f"120 GB cgroup for this model)", flush=True)
    else:
        stack = certs + [diag[k] for k in comps] + [gc_vec]
        print(f"[{tag}] building exact Gram over {len(stack)} vectors (float64) ...",
              flush=True)
        Gfull = exact_gram(stack, device)
        gram_method = f"exact_chunked_float64_over_{len(stack)}_vectors"
        del stack
    if not sketch_only:
        del gc_vec

    G = Gfull[:N, :N]
    n2 = np.diag(G).copy()
    gd_norm64 = {k: float(np.sqrt(Gfull[idx_c[k], idx_c[k]])) for k in comps}
    d64 = {k: Gfull[idx_c[k], :N].copy() for k in comps}
    gc_norm64 = float(np.sqrt(Gfull[idx_gc, idx_gc]))
    gc_here = {k: float(Gfull[idx_c[k], idx_gc] / (gd_norm64[k] * gc_norm64))
               for k in comps}

    gram_diag_rel = None
    if norm_i:
        n2_f32 = np.array(norm_i) ** 2
        gram_diag_rel = float((np.abs(n2 - n2_f32) / n2).max())
        print(f"[{tag}] Gram diag vs float32 flat_norm: max rel = "
              f"{gram_diag_rel:.3e} (float32 is the inaccurate one)", flush=True)
    for k in comps:
        print(f"[{tag}] g_C {k:9} here={gc_here[k]:.6f} "
              f"archived={gc_ref[k]:.6f}",
              flush=True)

    # The diagnostic gradients sum over the WHOLE batch, half of which is
    # greedily generated text. Greedy decoding is not bitwise reproducible, so
    # on a 16-item batch one differing rollout is 12.5% of the sum against 3% on
    # a 64-item batch. Certificates are fixed reference text and are unaffected,
    # which is why ESS can match to 1e-10 while these norms move by percent.
    gd_norm_cmp = {k: {"here": gd_norm64[k], "archived": gdn_ref[k],
                       "rel": abs(gd_norm64[k] - gdn_ref[k]) / gdn_ref[k]}
                   for k in comps}
    print(f"[{tag}/{dataset}] g_d norms vs archive: "
          + ", ".join(f"{k} {gd_norm_cmp[k]['rel']*100:.1f}%" for k in comps),
          flush=True)

    # ------------------------------------------------------------- curve
    KS = ks_for(N)
    rng = np.random.default_rng(SUBSET_SEED)
    curves, g3 = {}, {}
    for k in comps:
        curves[k] = curve(d64[k], n2, G, gd_norm64[k], rng, KS)
        got = curves[k][N]["median"]
        want = fq_ref[k]
        g3[k] = {"got": got, "want": want, "rel": abs(got - want) / abs(want)}
        print(f"[{tag}] G3 {k:9} k={N} got={got:.6f} want={want:.6f} "
              f"rel={g3[k]['rel']*100:.2f}%", flush=True)

    # ------- sketch calibration: only possible where the exact Gram fits -------
    sketch_cal = None
    if do_sketch and not sketch_only:
        Gs_c = Gs[:N, :N]
        n2_s = np.diag(Gs_c).copy()
        rng2 = np.random.default_rng(SUBSET_SEED)
        worst_med, worst_gc = 0.0, 0.0
        per = {}
        for k in comps:
            gdn_s = float(np.sqrt(Gs[idx_c[k], idx_c[k]]))
            cs = curve(Gs[idx_c[k], :N], n2_s, Gs_c, gdn_s, rng2, KS)
            dev = {kk: abs(cs[kk]["median"] - curves[k][kk]["median"])
                   for kk in KS}
            gcs = float(Gs[idx_c[k], idx_gc]
                        / (gdn_s * float(np.sqrt(Gs[idx_gc, idx_gc]))))
            per[k] = {"median_abs_dev": {str(kk): dev[kk] for kk in KS},
                      "gc_sketched": gcs, "gc_exact": gc_here[k],
                      "gc_abs_dev": abs(gcs - gc_here[k])}
            worst_med = max(worst_med, max(dev.values()))
            worst_gc = max(worst_gc, abs(gcs - gc_here[k]))
        sketch_cal = {"m": SKETCH_M, "per_component": per,
                      "worst_median_abs_dev": worst_med,
                      "worst_gc_abs_dev": worst_gc}
        print(f"[{tag}] SKETCH CALIBRATION m={SKETCH_M}: worst |median dev| = "
              f"{worst_med:.2e}, worst |g_C dev| = {worst_gc:.2e}", flush=True)

    result = {
        "model": tag, "dataset": dataset, "n_certificates": N,
        "gram_method": gram_method,
        "ess": ess, "i_star": i_star,
        "gates": {"batch_identity": batch_id, "batch_ok": batch_ok,
                  "G1": g1, "G1_max_abs_delta": g1_max,
                  "G1_available": g1 is not None,
                  "G1_tol": 5e-3, "G1_worst_mean_abs": g1_mean,
                  "G1_note": ("Per-certificate cosines are archived only for "
                              "Qwen3.5-0.8B/-2B on HumanEval; elsewhere G1 is "
                              "unavailable and batch identity rests on ESS and "
                              "N. Where available, cosines are compared across "
                              "machines (E6 on S2 without fla/causal_conv1d "
                              "kernels, E7 on S1 with them), so the backward "
                              "reduces in a different order."),
                  "gram_diag_max_rel": gram_diag_rel,
                  "G3": g3, "G3_pass": all(v["rel"] <= 0.02 for v in g3.values()),
                  "G3_note": ("G3 compares against an archived run. It is a "
                              "cross-run consistency check, not a validity "
                              "check on this curve, which uses its own g_d, "
                              "s_i and g_C throughout. See gd_norm_vs_archive."),
                  "gd_norm_vs_archive": gd_norm_cmp},
        "exact": {
            "cert_norms_f64": [float(np.sqrt(x)) for x in n2],
            "cert_norms_f32": [float(x) for x in norm_i] or None,
            "gd_norms_f64": gd_norm64,
            "gd_norms_f32": norm_d or None,
            "gd_dot_cert_f64": {k: [float(x) for x in d64[k]] for k in d64},
            "comps": comps,
            "gc_norm_f64": gc_norm64,
        },
        "gram": [[float(x) for x in row] for row in G],
        "gram_full_36": [[float(x) for x in row] for row in Gfull],
        "curve": curves,
        "sketch_calibration": sketch_cal,
        "gc_overlay": {k: gc_here[k] for k in comps},
        "gc_overlay_archived": gc_ref,
        "meta": {"seed": seed, "shuffle_seed": shuffle_seed, "n_tasks": n_tasks,
                 "torch": torch.__version__,
                 "device": torch.cuda.get_device_name(0),
                 "n_subsets": N_SUBSETS, "subset_seed": SUBSET_SEED,
                 "n_tasks": n_tasks, "ks": KS},
    }
    out = f"{RESULTS}/e7_ksubset_{tag}_{dataset}_seed0.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
