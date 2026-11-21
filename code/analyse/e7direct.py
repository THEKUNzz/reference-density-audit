"""Direct test of the claim Codex objected to.

Three equal component cosines do not establish that g_C equals the dominant
certificate's gradient. But the full Gram contains the angle between them
directly, so measure it instead of inferring it.
"""
import json, glob, os
import numpy as np
os.chdir('/tmp/e7src')
rows = []
for f in sorted(glob.glob('e7_ksubset_*_seed0.json')):
    d = json.load(open(f))
    N = d['n_certificates']; istar = d['i_star']; ess = d['ess']
    Gf = np.array(d['gram_full_36'])
    comps = d['exact'].get('comps') or ['hsd', 'opid', 'edge_opd']
    igc = N + len(comps)
    cos_gc_istar = float(Gf[igc, istar] / np.sqrt(Gf[igc, igc] * Gf[istar, istar]))
    # how much of g_C's norm is NOT along s_i*
    frac = float(np.sqrt(max(1 - cos_gc_istar**2, 0)))
    rows.append((d['model'], d['dataset'], ess, cos_gc_istar, frac, d['gram_method'][:5]))
print("{:<16}{:<11}{:>6}{:>16}{:>14}{:>7}".format('model','dataset','ESS','cos(g_C,s_i*)','orthog. frac','gram'))
for r in sorted(rows, key=lambda x: x[2]):
    print("{:<16}{:<11}{:>6.2f}{:>16.5f}{:>14.4f}{:>7}".format(*r))
e1 = [r for r in rows if round(r[2], 2) == 1.00]
print("\nESS == 1.00 cells (n={}): cos(g_C, s_i*) min = {:.5f}, max = {:.5f}".format(
    len(e1), min(r[3] for r in e1), max(r[3] for r in e1)))
lo = [r for r in rows if r[2] < 1.3]
print("ESS <  1.30 cells (n={}): cos(g_C, s_i*) min = {:.4f}".format(len(lo), min(r[3] for r in lo)))
hi = [r for r in rows if r[2] >= 1.8]
print("ESS >= 1.80 cells (n={}): cos(g_C, s_i*) ranges {:.4f} - {:.4f}".format(
    len(hi), min(r[3] for r in hi), max(r[3] for r in hi)))
