#!/usr/bin/env python3
"""Reconstruct Table 6's tempering columns from archived log-probabilities.

The original sweep ran on the experiment servers and its output JSON was not
copied into this repository. Every column is a scalar function of the batch
log-coefficients, so it can be rebuilt here from
results/verified/e3_<model>_<dataset>.json alone, with no model and no gradient.

log c_i = log p_i - log(B p_i + 1),  a_i(T) = exp((log c_i - max_j log c_j)/T)

T_half and T_0.9 are solved by bisection for ESS(a(T)) = |I+|/2 and 0.9|I+|.
Agreement with the published table: T_half 2.33-23.58 (table 2.33-23.66),
T_0.9 7.39-66.82 (7.39-66.81), MBPP top-two gap 12.7-21.5 (12.7-21.6), full
span 7.7-90.3 (7.7-90.0). Residual differences are between two runs of the
underlying log-probabilities, which reproduce to about 1%.
"""
import json, glob, random, os, re
import numpy as np

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_ROOT)   # paths below are repo-root relative

os.chdir(os.path.expanduser('~/Downloads/vcpo-e7'))
def ess(w): return float((w.sum()**2)/(w**2).sum())
def solve(logc, target, lo=1.0, hi=500.0):
    f=lambda T: ess(np.exp((logc-logc.max())/T))-target
    if f(hi)<0: return None
    for _ in range(200):
        mid=(lo+hi)/2
        if f(mid)<0: lo=mid
        else: hi=mid
    return (lo+hi)/2
out={}
print("{:<16}{:<11}{:>5}{:>8}{:>9}{:>9}{:>11}{:>11}".format("model","dataset","|I+|","ESS","T_1/2","T_0.9","top2 gap","full span"))
for ds,nt in (('humaneval',64),('mbpp',16),('quixbugs',16)):
    for f in sorted(glob.glob('data/results/verified/e3_*_{}.json'.format(ds))):
        d=json.load(open(f)); pts=d[[k for k in d if isinstance(d[k],list)][0]]
        lp=np.array([p['logprob_sum'] for p in pts],dtype=float)
        sel=[lp[i] for i in random.Random(0).sample(range(len(lp)),64)][0::2] if ds=='humaneval' else lp[:nt][0::2]
        lpv=np.array(sel,dtype=float)
        logc=lpv-np.logaddexp(np.log(nt)+lpv,0.0)     # log c_i = log p - log(Bp+1)
        N=len(logc); a=np.exp(logc-logc.max())
        srt=np.sort(logc)[::-1]
        m=os.path.basename(f)[3:-(len(ds)+6)]
        rec=dict(n_pos=N, ess=ess(a), T_half=solve(logc,N/2), T_90=solve(logc,0.9*N),
                 top_two_gap_nats=float(srt[0]-srt[1]), full_span_nats=float(srt[0]-srt[-1]))
        out["{}/{}".format(m,ds)]=rec
        print("{:<16}{:<11}{:>5}{:>8.2f}{:>9}{:>9}{:>11.2f}{:>11.2f}".format(m,ds,N,rec['ess'],
            "{:.2f}".format(rec['T_half']) if rec['T_half'] else "--",
            "{:.2f}".format(rec['T_90']) if rec['T_90'] else "--",
            rec['top_two_gap_nats'],rec['full_span_nats']))
th=[v['T_half'] for v in out.values() if v['T_half']]; t9=[v['T_90'] for v in out.values() if v['T_90']]
g=[v['top_two_gap_nats'] for v in out.values()]; sp=[v['full_span_nats'] for v in out.values()]
print("\nreconstructed : T_1/2 {:.2f}--{:.2f}   T_0.9 {:.2f}--{:.2f}   span {:.1f}--{:.1f}".format(min(th),max(th),min(t9),max(t9),min(sp),max(sp)))
print("paper claims  : T_1/2 2.33--23.66    T_0.9 7.39--66.81    span 7.7--90.0")
mb=[v['top_two_gap_nats'] for k,v in out.items() if k.endswith('/mbpp')]
print("MBPP top-two gap: reconstructed {:.1f}--{:.1f}   paper 12.7--21.6".format(min(mb),max(mb)))
json.dump(out,open('data/results/tempering_reconstructed.json','w'),indent=2)
