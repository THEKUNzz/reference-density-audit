#!/usr/bin/env python3
"""Two controls separating length from the multiplicative accumulation itself.

Both read only archived teacher-forced reference log-probabilities
(results/verified/e3_<model>_<dataset>.json, points[].{logprob_sum,
logprob_mean_per_token, n_tokens}). No model is loaded and no gradient is taken.

Control 1 -- length-matched windows.  Sort the full reference pool by token
count and take every contiguous window of 8. Within a window the longest
certificate is at most ~1.5x the shortest, so length is held nearly fixed.
Report the median ESS over windows and how often the shortest certificate
carries the largest weight (chance = 1/8 = 0.125).

Control 2 -- per-token normalisation.  Reweight the same pool by
exp(logprob_sum / n_tokens) = p_theta^(1/L), the per-token geometric mean,
and report ESS/N. NOTE this divides the whole log-probability by L, removing
the length scale from the residual as well; it shows that compressing the
scale restores coverage, not that length caused the collapse.

N here is the full reference pool (164 / 257 / 40), not the 8- or 32-reference
diagnostic batch of Table 6.
"""
import json, glob, os
import numpy as np

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_ROOT)   # paths below are repo-root relative


WINDOW = 8
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "length_controls.json")

def ess(w):
    return float((w.sum() ** 2) / (w ** 2).sum())

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    hdr = "{:<16}{:<11}{:>5}{:>9}{:>12}{:>11}{:>12}{:>12}"
    print(hdr.format("model", "dataset", "N", "windows", "med ESS(win)",
                     "P(short)", "ESS/N seq", "ESS/N 1/L"))
    for ds in ("humaneval", "mbpp", "quixbugs"):
        for f in sorted(glob.glob(os.path.join(root, "data/results/verified",
                                               "e3_*_{}.json".format(ds)))):
            d = json.load(open(f))
            pts = d[[k for k in d if isinstance(d[k], list)][0]]
            lp = np.array([p["logprob_sum"] for p in pts], dtype=float)
            L = np.array([p["n_tokens"] for p in pts], dtype=float)
            pt = np.array([p["logprob_mean_per_token"] for p in pts], dtype=float)
            model = os.path.basename(f)[3:-(len(ds) + 6)]

            o = np.argsort(L); lpw, Lw = lp[o], L[o]
            e, short, spread = [], [], []
            for i in range(len(Lw) - WINDOW + 1):
                seg = lpw[i:i + WINDOW]
                a = np.exp(seg - seg.max())
                e.append(ess(a))
                short.append(int(np.argmax(a) == 0))
                spread.append(float(Lw[i + WINDOW - 1] / max(Lw[i], 1)))

            w_seq = np.exp(lp - lp.max())
            w_pt = np.exp(pt - pt.max())
            rec = dict(n_pool=len(lp), n_windows=len(e), window=WINDOW,
                       median_ess_window=float(np.median(e)),
                       p_shortest_wins=float(np.mean(short)),
                       chance=1.0 / WINDOW,
                       median_length_spread_in_window=float(np.median(spread)),
                       ess_over_n_sequence=ess(w_seq) / len(lp),
                       ess_over_n_per_token=ess(w_pt) / len(lp))
            out["{}/{}".format(model, ds)] = rec
            print(hdr.format(model, ds, len(lp), len(e),
                             "{:.2f}".format(rec["median_ess_window"]),
                             "{:.3f}".format(rec["p_shortest_wins"]),
                             "{:.3f}".format(rec["ess_over_n_sequence"]),
                             "{:.3f}".format(rec["ess_over_n_per_token"])))
    json.dump(out, open(OUT, "w"), indent=2)
    mw = [v["median_ess_window"] for v in out.values()]
    ps = [v["p_shortest_wins"] for v in out.values()]
    sq = [v["ess_over_n_sequence"] for v in out.values()]
    pr = [v["ess_over_n_per_token"] for v in out.values()]
    print("\nmedian ESS in length-matched windows : {:.2f}-{:.2f}".format(min(mw), max(mw)))
    print("P(shortest wins)                     : {:.2f}-{:.2f}  (chance 0.125)".format(min(ps), max(ps)))
    print("ESS/N, sequence probability          : {:.3f}-{:.3f}".format(min(sq), max(sq)))
    print("ESS/N, per-token geometric mean      : {:.3f}-{:.3f}".format(min(pr), max(pr)))
    print("\nwrote", os.path.relpath(OUT, root))

if __name__ == "__main__":
    main()
