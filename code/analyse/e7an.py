import json, glob, os
import numpy as np
os.chdir('/tmp/e7src')
NAME = {'hsd': 'TeachCE', 'opid': 'LogShift', 'edge_opd': 'MaskOnly'}
rows = []
for f in sorted(glob.glob('e7_ksubset_*_seed0.json')):
    d = json.load(open(f))
    G = np.array(d['gram']); n2 = np.diag(G)
    ex = d['exact']; ess = d['ess']; N = d['n_certificates']
    ks = np.array(sorted(int(k) for k in d['curve']['hsd']), dtype=float)
    cell = d['model'].replace('Qwen3.5-', 'Q').replace('OLMo-2-1124-', 'O')
    for c in ('hsd', 'opid', 'edge_opd'):
        cv = d['curve'][c]; gc = d['gc_overlay'][c]
        singles = np.array(ex['gd_dot_cert_f64'][c]) / (ex['gd_norms_f64'][c] * np.sqrt(n2))
        med = np.array([cv[str(int(k))]['median'] for k in ks])
        lo = cv[str(int(ks[0]))]['p05']; hi = cv[str(int(ks[0]))]['p95']
        rows.append(dict(cell=cell + '/' + d['dataset'][:2].upper(), comp=NAME[c],
                         N=N, ess=ess, gc=gc,
                         pct=float(np.mean(singles <= gc)) * 100,
                         keff=float(np.interp(gc, med, ks)),
                         inside=bool(lo <= gc <= hi),
                         ratio=med[-1] / med[0],
                         g3=d['gates']['G3'][c]['rel'] * 100))
hdr = "{:<12}{:<9}{:>3}{:>6}{:>8}{:>8}{:>7}{:>7}{:>6}{:>6}"
print(hdr.format('cell', 'comp', 'N', 'ESS', 'g_C', 'pct@k1', 'k_eff', 'kN/k1', 'band', 'G3%'))
for r in rows:
    print("{:<12}{:<9}{:>3}{:>6.2f}{:>8.4f}{:>7.1f}%{:>7.2f}{:>7.2f}{:>6}{:>6.1f}".format(
        r['cell'], r['comp'], r['N'], r['ess'], r['gc'], r['pct'], r['keff'],
        r['ratio'], 'in' if r['inside'] else 'OUT', r['g3']))
n = len(rows)
print("\n=== {} measurements, {} cells ===".format(n, n // 3))
print("g_C inside the k=1 band  : {}/{}".format(sum(r['inside'] for r in rows), n))
print("k_eff <= ESS + 1         : {}/{}".format(sum(r['keff'] <= r['ess'] + 1 for r in rows), n))
print("k_eff range              : {:.2f} - {:.2f}".format(min(r['keff'] for r in rows), max(r['keff'] for r in rows)))
print("aggregation gain kN/k1   : {:.2f}x - {:.2f}x".format(min(r['ratio'] for r in rows), max(r['ratio'] for r in rows)))
print("G3 within 2%             : {}/{}  (worst {:.1f}%)".format(sum(r['g3'] <= 2 for r in rows), n, max(r['g3'] for r in rows)))
print("\n--- discriminating cell: Q9B/HU, ESS 3.13, theory predicts k_eff ~ 3 ---")
for r in rows:
    if r['cell'] == 'Q9B/HU':
        print("  {:<9} ESS={:.2f}  k_eff={:.2f}  pct@k1={:.1f}%".format(r['comp'], r['ess'], r['keff'], r['pct']))
