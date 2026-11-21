"""Does the supervised chain hold across models?

Chain link 3: at ESS ~ 1, g_C reduces to ONE certificate's gradient, which is
that certificate's supervised gradient. Test it directly: compare
cos(g_d, g_C) against cos(g_d, s_{i*}) where i* is the dominant reference.
This is independent of the k-subset curve.
"""
import json, glob, os
import numpy as np
os.chdir('/tmp/e7src')
NAME = {'hsd': 'TeachCE', 'opid': 'LogShift', 'edge_opd': 'MaskOnly'}
rows = []
for f in sorted(glob.glob('e7_ksubset_*_seed0.json')):
    d = json.load(open(f))
    G = np.array(d['gram']); n2 = np.diag(G)
    ex = d['exact']; istar = d['i_star']; ess = d['ess']
    cell = d['model'].replace('Qwen3.5-', 'Q').replace('OLMo-2-1124-', 'O') + '/' + d['dataset'][:2].upper()
    for c in ('hsd', 'opid', 'edge_opd'):
        gd_n = ex['gd_norms_f64'][c]
        dd = np.array(ex['gd_dot_cert_f64'][c])
        cos_istar = float(dd[istar] / (gd_n * np.sqrt(n2[istar])))
        gc = d['gc_overlay'][c]
        rel = abs(gc - cos_istar) / abs(cos_istar) if cos_istar else float('nan')
        rows.append((cell, NAME[c], ess, gc, cos_istar, rel * 100))
print("{:<11}{:<9}{:>6}{:>10}{:>12}{:>10}".format('cell', 'comp', 'ESS', 'cos(gd,gC)', 'cos(gd,s_i*)', 'rel diff'))
for r in rows:
    print("{:<11}{:<9}{:>6.2f}{:>10.4f}{:>12.4f}{:>9.1f}%".format(*r))
rels = [r[5] for r in rows if np.isfinite(r[5])]
low = [r for r in rows if r[2] < 1.3]
print("\nAll {} measurements: median |rel diff| = {:.1f}%, max = {:.1f}%".format(len(rels), np.median(rels), max(rels)))
print("Cells with ESS < 1.3 (where g_C SHOULD equal one certificate):")
print("   n={}, median = {:.1f}%, max = {:.1f}%".format(
    len(low), np.median([r[5] for r in low]), max(r[5] for r in low)))
