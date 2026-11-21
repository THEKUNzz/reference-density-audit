import json, glob, random, os
import numpy as np

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_ROOT)   # paths below are repo-root relative

os.chdir(os.path.expanduser('~/Downloads/vcpo-e7'))
out={}
for ds,nt in (('humaneval',64),('mbpp',16),('quixbugs',16)):
    for f in sorted(glob.glob('data/results/verified/e3_*_{}.json'.format(ds))):
        d=json.load(open(f)); pts=d[[k for k in d if isinstance(d[k],list)][0]]
        lp=np.array([p['logprob_sum'] for p in pts],dtype=float)
        sel=[lp[i] for i in random.Random(0).sample(range(len(lp)),64)][0::2] if ds=='humaneval' else lp[:nt][0::2]
        p=np.exp(np.array(sel,dtype=float))
        ess=lambda w:float((w.sum()**2)/(w**2).sum())
        a0=p/p.max(); c=p/(nt*p+1); a1=c/c.max()
        m=os.path.basename(f)[3:-(len(ds)+6)]
        out["{}/{}".format(m,ds)]=dict(B=nt,n_certificates=len(p),Bp_max=float(nt*p.max()),
            ess_weights_p=ess(a0),ess_with_balance_denominator=ess(a1),
            rel_change=float(ess(a1)/ess(a0)-1))
json.dump(out,open('data/results/denominator_effect.json','w'),indent=2)
bp=[v['Bp_max'] for v in out.values()]; ch=[v['rel_change'] for v in out.values()]
print("B*p_max over the 18 diagnostic batches : {:.1e} .. {:.2f}".format(min(bp),max(bp)))
print("ESS change when the denominator is kept: {:+.1%} .. {:+.1%}".format(min(ch),max(ch)))
for k,v in sorted(out.items(),key=lambda x:-x[1]['rel_change'])[:3]:
    print("   largest: {:<26} Bp_max={:.2f}  ESS {:.2f} -> {:.2f}".format(k,v['Bp_max'],v['ess_weights_p'],v['ess_with_balance_denominator']))
print("wrote results/denominator_effect.json")
