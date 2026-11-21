"""Recover gc_overlay from the stored full Gram, and validate against the
cells where it was written correctly."""
import json, glob, os
import numpy as np
os.chdir('/tmp/e7src')
ok, fixed = [], []
for f in sorted(glob.glob('e7_ksubset_*_seed0.json')):
    d = json.load(open(f))
    N = d['n_certificates']
    Gf = np.array(d['gram_full_36'])
    comps = d['exact'].get('comps') or ['hsd', 'opid', 'edge_opd']
    idx_gc = N + len(comps)
    rec = {}
    for i, c in enumerate(comps):
        ic = N + i
        rec[c] = float(Gf[ic, idx_gc] / (np.sqrt(Gf[ic, ic]) * np.sqrt(Gf[idx_gc, idx_gc])))
    stored = d.get('gc_overlay') or {}
    label = "{}/{}".format(d['model'], d['dataset'])
    if stored:
        dev = max(abs(rec[c] - stored[c]) for c in comps)
        ok.append((label, dev))
    else:
        d['gc_overlay'] = rec
        d['gc_overlay_recovered_from_gram'] = True
        json.dump(d, open(f, 'w'), indent=2)
        fixed.append((label, rec))

print("VALIDATION -- recomputed vs stored, on cells that have both:")
for label, dev in ok:
    print("  {:<30} max |dev| = {:.3e}".format(label, dev))
print("\nRECOVERED -- cells where gc_overlay was empty:")
for label, rec in fixed:
    print("  {:<30} ".format(label) + "  ".join("{}={:.4f}".format(c, v) for c, v in rec.items()))
