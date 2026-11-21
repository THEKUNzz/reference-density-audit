"""Recompute paper summaries from archived JSON; no model loading or experiments."""
from pathlib import Path
import hashlib, json, math, statistics
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/results/verified'
MODELS = ['Qwen3.5-0.8B','Qwen3.5-2B','Qwen3.5-4B','Qwen3.5-9B','OLMo-2-1124-7B','glm-4-9b']
def read(name): return json.loads((DATA/name).read_text())
def mean(x): return statistics.mean(x)
def interp(curve, target):
    # Only valid after monotonicity has been checked.
    for a,b in zip(curve,curve[1:]):
        if a['train_fit'] <= target <= b['train_fit']:
            t=(target-a['train_fit'])/(b['train_fit']-a['train_fit'])
            return {k:a[k]+t*(b[k]-a[k]) for k in ['train_fit','heldout_fit','step']}
    raise ValueError('target outside curve')
def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]
out={'training':[],'allfail':[],'alignment':[],'decay':[],'mismatch':[]}
for m in MODELS:
    for ds in ['humaneval','mbpp','quixbugs']:
        e3=read(f'e3_{m}_{ds}.json'); logs=[p['logprob_sum'] for p in e3['points']]
        w=[math.exp(x-max(logs)) for x in logs]
        out['decay'].append({'model':m,'dataset':ds,'n':len(logs),'token':e3['fit_per_token'],'char':e3['fit_per_char'],'probability_ess':sum(w)**2/sum(x*x for x in w),'horizon':read(f'e5_{m}_{ds}.json')['report_lengths']})
        suf={'humaneval':'_shuf64','mbpp':'_mbpp','quixbugs':'_quixbugs'}[ds]
        e6=read(f'e6_{m}{suf}.json')
        out['alignment'].append({'model':m,'dataset':ds,**e6})
        e2c=read(f'e2c_{m}{"" if ds=="humaneval" else "_"+ds}.json')
        out['mismatch'].append({'model':m,'dataset':ds,**e2c})
        if ds=='quixbugs': continue
        e1=[read(f'e1formal_{m}_seed{i}{"_mbpp" if ds=="mbpp" else ""}.json') for i in range(3)]
        k=sum(d['n_all_fail'] for d in e1); n=sum(d['n_groups'] for d in e1)
        representative_group_checks=sum(d['grad_vcpo_norm'] is not None for d in e1)
        individual_gradient_norms=sum(
            int(d['grad_grpo_norm'] is not None)+int(d['grad_vcpo_norm'] is not None)
            for d in e1
        )
        out['allfail'].append({'model':m,'dataset':ds,'allfail':k,'groups':n,'rate':k/n,'wilson':wilson(k,n),'representative_group_checks':representative_group_checks,'individual_gradient_norms':individual_gradient_norms,'checks_nonzero':sum((d['grad_vcpo_norm'] or 0)>0 for d in e1),'checks_grpo_zero':sum(d['grad_grpo_norm']==0 for d in e1),'harness_fields_present':sum('n_harness_error' in d for d in e1),'harness_errors':sum(d.get('n_harness_error',0) for d in e1)})
        if m=='glm-4-9b' or (m.startswith('OLMo') and ds=='mbpp'):continue
        runs=[]
        for i in range(1 if m.startswith('OLMo') else 3):
            folder='kernel/' if ds=='mbpp' or m=='Qwen3.5-9B' else ''
            fn=f'{folder}e2b_{m}_seed{i}{"_mbpp" if ds=="mbpp" else ""}.json'; d=read(fn)
            assert all(d['curves'][a][0]['step']==0 and d['curves'][a][-1]['step']==300 for a in ['fq','vcpo'])
            moves={a:d['curves'][a][-1]['train_fit']-d['curves'][a][0]['train_fit'] for a in ['fq','vcpo']}
            mono={a:all(y['train_fit']>=x['train_fit'] for x,y in zip(d['curves'][a],d['curves'][a][1:])) for a in ['fq','vcpo']}
            row={'file':fn,'seed':i,'gap_difference':d['gap_fq']-d['gap_vcpo'],'moves':moves,'monotone':mono,'N':d['n_R1']+d['n_R0']}
            if all(mono.values()):
                base=max(d['curves'][a][0]['train_fit'] for a in ['fq','vcpo'])
                target=base+.9*(min(d['curves'][a][-1]['train_fit'] for a in ['fq','vcpo'])-base)
                matched={a:interp(d['curves'][a],target) for a in ['fq','vcpo']}
                row['fit_match_90']={'target':target,'gap_difference':matched['vcpo']['heldout_fit']-matched['fq']['heldout_fit'],'steps':{a:v['step'] for a,v in matched.items()}}
            runs.append(row)
        moves={a:mean([r['moves'][a] for r in runs]) for a in ['fq','vcpo']}
        diffs=[r['gap_difference'] for r in runs]
        out['training'].append({'model':m,'dataset':ds,'runs':runs,'mean':mean(diffs),'sd':statistics.stdev(diffs) if len(diffs)>1 else None,'moves':moves,'movement_ratio':moves['fq']/moves['vcpo'],'match90':mean([r['fit_match_90']['gap_difference'] for r in runs]) if all('fit_match_90' in r for r in runs) else None})
out['manifest']=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(DATA.rglob('*.json'))]
path=ROOT/'data/RESULTS_AUDIT.json';path.write_text(json.dumps(out,indent=2)+'\n')
print('Saved',path)
for r in out['training']:print(r['model'],r['dataset'],r['mean'],r['moves'],r['movement_ratio'],'match90',r['match90'])
