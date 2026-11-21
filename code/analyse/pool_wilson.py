#!/usr/bin/env python3
"""Pool E1 seeds per model and report a Wilson interval for the all-fail rate.

Wilson rather than the normal approximation: GLM has few all-fail groups and
p +/- 1.96*SE is unreliable close to the boundary. Each file is processed once
(dedup), and only *_seed*.json files are pooled so stale pre-seed files are
excluded.
"""
import json, glob
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - half), min(1.0, centre + half)


paths = set(glob.glob("e1formal_*_seed*.json"))
pooled = defaultdict(lambda: [0, 0, []])
for path in sorted(paths):
    d = json.load(open(path))
    extra = d["meta"]["extra"]
    frac = extra.get("all_fail_fraction")
    n = extra.get("n_tasks")
    if frac is None or not n:
        continue
    name = d["meta"]["model"].split("/")[-1]
    rec = pooled[name]
    rec[0] += round(frac * n)
    rec[1] += n
    rec[2].append((d["meta"]["seed"], frac))

print(f"{'model':22s} {'seeds':>26s} {'pooled':>14s} {'95% CI (Wilson)':>20s}")
for name, (k, n, per_seed) in sorted(pooled.items()):
    lo, hi = wilson(k, n)
    seeds = " ".join(f"s{s}:{f:.2f}" for s, f in sorted(per_seed))
    print(f"{name:22s} {seeds:>26s} {k:3d}/{n:3d}={k/n:6.3f}  [{lo:.3f}, {hi:.3f}]")
