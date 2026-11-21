import json, glob, os, csv
import numpy as np
os.chdir('/tmp/e7src')
NAME = {'hsd': 'teacher CE', 'opid': 'log-shift', 'edge_opd': 'mask-only'}
DS = {'humaneval': 'HE', 'mbpp': 'MBPP', 'quixbugs': 'QB'}
rows = []
for f in sorted(glob.glob('e7_ksubset_*_seed0.json')):
    d = json.load(open(f))
    G = np.array(d['gram']); n2 = np.diag(G); ex = d['exact']
    N = d['n_certificates']; ess = d['ess']; istar = d['i_star']
    ks = np.array(sorted(int(k) for k in d['curve']['hsd']), dtype=float)
    for c in ('hsd', 'opid', 'edge_opd'):
        cv = d['curve'][c]; gc = d['gc_overlay'][c]
        gdn = ex['gd_norms_f64'][c]; dd = np.array(ex['gd_dot_cert_f64'][c])
        singles = dd / (gdn * np.sqrt(n2))
        med = np.array([cv[str(int(k))]['median'] for k in ks])
        rows.append(dict(model=d['model'], dataset=DS[d['dataset']], comp=NAME[c],
            N=N, ess=round(ess, 3), k1_med=round(float(med[0]), 4),
            kN=round(float(med[-1]), 4), gain=round(float(med[-1] / med[0]), 2),
            gc=round(float(gc), 4),
            gc_pct_in_k1=round(float(np.mean(singles <= gc)) * 100, 1),
            k_eff=round(float(np.interp(gc, med, ks)), 2),
            in_k1_band=bool(cv[str(int(ks[0]))]['p05'] <= gc <= cv[str(int(ks[0]))]['p95']),
            cos_istar=round(float(dd[istar] / (gdn * np.sqrt(n2[istar]))), 4),
            cos_gc_istar=round(float(np.array(d['gram_full_36'])[N+3, istar] /
                np.sqrt(np.array(d['gram_full_36'])[N+3,N+3] * np.array(d['gram_full_36'])[istar,istar])), 5),
            G3_rel_pct=round(d['gates']['G3'][c]['rel'] * 100, 2),
            gram=d['gram_method'].split('_')[0]))
with open('/tmp/e7src/E7_ALL_CELLS.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote E7_ALL_CELLS.csv:", len(rows), "rows,", len(rows)//3, "cells")
g = [r['gain'] for r in rows]
print("aggregation gain  : {:.2f}x - {:.2f}x  (all {} measurements)".format(min(g), max(g), len(rows)))
print("g_C in k=1 band   : {}/{}".format(sum(r['in_k1_band'] for r in rows), len(rows)))
print("k_eff             : {:.2f} - {:.2f}".format(min(r['k_eff'] for r in rows), max(r['k_eff'] for r in rows)))
e1 = [r for r in rows if r['ess'] == 1.0]
dev = [abs(r['gc'] - r['cos_istar']) for r in e1]
print("ESS==1.00 cells   : {} measurements, max |cos(gd,gC) - cos(gd,s_i*)| = {:.1e}".format(len(e1), max(dev)))
lo = [r for r in rows if r['ess'] < 1.3]
rel = [abs(r['gc'] - r['cos_istar']) / abs(r['cos_istar']) * 100 for r in lo]
print("ESS < 1.3         : {} measurements, median rel dev {:.1f}%, max {:.1f}%".format(len(lo), np.median(rel), max(rel)))
print("G3 <= 2%          : {}/{}, worst {:.1f}%".format(sum(r['G3_rel_pct'] <= 2 for r in rows), len(rows), max(r['G3_rel_pct'] for r in rows)))
print("exact Gram cells  :", sum(1 for r in rows if r['gram'] == 'exact') // 3, "of", len(rows)//3)
