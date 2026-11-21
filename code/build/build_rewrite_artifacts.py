"""Build fresh manuscript tables from archived measurements; never edit the old paper."""
from pathlib import Path
import json, statistics, hashlib
ROOT=Path(__file__).resolve().parents[2]
# Recompute all reported summaries using the existing audited extraction logic.
code=(ROOT/'scripts/audit_paper_results.py').read_text().replace("'paper/RESULTS_AUDIT.json'", "'paper/REWRITE_RESULTS_AUDIT.json'")
exec(compile(code, 'audit_paper_results.py', 'exec'), {'__file__':str(ROOT/'scripts/audit_paper_results.py')})
d=json.loads((ROOT/'paper/REWRITE_RESULTS_AUDIT.json').read_text())
temp=json.loads((ROOT/'tmp/e6_temperature_raw/e6_temperature_sweep.json').read_text())
tc=json.loads((ROOT/'tmp/e6_temperature_raw/e6_temperature_cosines.json').read_text())
mods=['Qwen3.5-0.8B','Qwen3.5-2B','Qwen3.5-4B','Qwen3.5-9B','OLMo-2-1124-7B','glm-4-9b']
names=dict(zip(mods,['Qwen 0.8B','Qwen 2B','Qwen 4B','Qwen 9B','OLMo 7B','GLM 9B']))
short=dict(zip(mods,['Q0.8','Q2','Q4','Q9','O7','G9']))
dsname={'humaneval':'HE','mbpp':'MBPP','quixbugs':'QB'}
ks=['hsd','opid','edge_opd'];kn=['Teacher CE','Log-shift','Mask-only']
def get(key,m,ds): return next(r for r in d[key] if r['model']==m and r['dataset']==ds)
def table(spec,head,rows,caption,label,wide=False):
 env='table*' if wide else 'table'
 return '\\begin{'+env+'}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+head+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{'+env+'}'
b={}
rows=[]
for m in [mods[4],mods[0],mods[1],mods[5],mods[2],mods[3]]:
 vals=[]
 for ds in ['humaneval','mbpp']:
  r=get('allfail',m,ds);lo,hi=r['wilson'];vals.append(f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}]")
 rows.append([names[m],*vals])
b['ALLFAIL_TABLE_NEW']=table('lcc','Model & HumanEval & MBPP',rows,r'All-fail rates and pooled 95\% Wilson intervals. Each cell contains 120 groups (40 tasks, three seeds), with four rollouts per group.','tab:allfail-new')
rows=[]
for k,nm in zip(ks,kn):
 vals=[r['alignment'][k]['cos_with_grad_Fq']-r['alignment'][k]['cos_with_grad_VCPO'] for r in d['alignment']]
 assert all(v>0 for v in vals)
 rows.append([nm,f'{sum(v>0 for v in vals)}/18',f'{min(vals):.3f}',f'{statistics.median(vals):.3f}',f'{max(vals):.3f}'])
b['MARGIN_TABLE_NEW']=table('lrrrr',r'Component & $\Delta>0$ & Min. & Median & Max.',rows,r'Cosine margins $\Delta=\cos(g_d,\gF)-\cos(g_d,\gC)$. Each component has six models $\times$ three datasets.','tab:margins-new')
rows=[]
for m in mods:
 vals=[]
 for ds in dsname:
  r=get('mismatch',m,ds); vals += [f"{r['cos_vcpo_clean_vs_corrupted']:.3f}",f"{r['cos_fq_clean_vs_corrupted']:.3f}"]
 rows.append([short[m],*vals])
b['PAIRING_TABLE_NEW']=table('lrrrrrr',r'& \multicolumn{2}{c}{HE} & \multicolumn{2}{c}{MBPP} & \multicolumn{2}{c}{QB}\\ Model & $\gC$ & $\gF$ & $\gC$ & $\gF$ & $\gC$ & $\gF$',rows,r'Clean/mismatched cosines with fixed labels, 32 tasks per cell. Model abbreviations retain family and size. Lower values indicate greater directional sensitivity.','tab:pair-new')
rows=[]
for ds in ['humaneval','mbpp']:
 for r in d['training']:
  if r['dataset']!=ds:continue
  sd=f" $\\pm$ {r['sd']:.4f}" if r['sd'] is not None else ' (one seed)'
  rows.append([dsname[ds],names[r['model']],f"${r['mean']:+.4f}$"+sd,f"{r['moves']['fq']:.4f}",f"{r['moves']['vcpo']:.4f}",f"{r['movement_ratio']:.2f}"])
b['TRAINING_TABLE_NEW']=table('lllrrr',r'Data & Model & Gap difference & Move: $F_q$ & Move: $\gC$ & Ratio',rows,r'Step-300 gap differences (nats/token), mean $\pm$ sample standard deviation over three seeds except OLMo. Positive values mean the likelihood arm has the larger train--held-out gap. Movements are final minus initial training fit; ratio divides their across-seed means.','tab:training-new',True)
rows=[]
for ds in dsname:
 for m in mods:
  r=get('alignment',m,ds);vals=[]
  for k in ks:
   a=r['alignment'][k];vals += [f"{a['cos_with_grad_Fq']:.3f}",f"{a['cos_with_grad_VCPO']:.3f}"]
  rows.append([dsname[ds],names[m],*vals])
b['FULL_ALIGNMENT_NEW']=table('llrrrrrr',r'& & \multicolumn{2}{c}{Teacher CE} & \multicolumn{2}{c}{Log-shift} & \multicolumn{2}{c}{Mask-only}\\ Data & Model & $\gF$ & $\gC$ & $\gF$ & $\gC$ & $\gF$ & $\gC$',rows,'Complete component cosines: 64-task HumanEval subset and 16-task MBPP/QuixBugs subsets, seed 0. GLM/HumanEval is retained, including its near-zero endpoint cosines.','tab:align-new',True)
rows=[]
for ds in dsname:
 for m in mods:
  r=next(x for x in temp if x['model']==m and x['dataset']==ds)
  rows.append([dsname[ds],names[m],str(r['N']),f"{r['ess_vs_T']['1']:.2f}",f"{r['T_half']:.2f}",f"{r['T_90']:.2f}",f"{r['logc_spread_max_minus_second']:.2f}",f"{r['logc_spread_max_minus_min']:.2f}"])
b['ESS_TABLE_NEW']=table('llrrrrrr',r'Data & Model & $|I^+|$ & ESS(1) & $T_{1/2}$ & $T_{0.9}$ & Top-two gap & Full span',rows,r'All 18 scalar weight sweeps. $T_{1/2}$ and $T_{0.9}$ reach half and 90\% of the positive reference count. Gaps and spans are in natural-log units. MBPP ESS values round to 1.00.','tab:ess-new',True)
rows=[]
for ds in dsname:
 for m in mods:
  r=get('decay',m,ds);rows.append([dsname[ds],names[m],str(r['n']),f"{r['token']['intercept']:.3f}",f"{r['token']['slope']:.3f}",f"{r['char']['slope']:.4f}",f"{r['token']['r2']:.2f}",f"{r['probability_ess']:.2f}"])
b['DECAY_TABLE_NEW']=table('llrrrrrr',r'Data & Model & Count & Intercept & /token & /char & $R^2$ & ESS',rows,r'Whole-reference $\log_{10}p_\theta$ OLS fits with intercepts and probability-weighted ESS on full datasets. $R^2$ is for the token regression. These values are distinct from diagnostic-batch ESS and general MIS coefficients.','tab:decay-new',True)
rows=[]
for r in d['training']:
 if r['dataset']!='humaneval':continue
 steps={a:statistics.mean(x['fit_match_90']['steps'][a] for x in r['runs']) for a in ['fq','vcpo']}
 rows.append([names[r['model']],f"${r['match90']:+.4f}$",f"{steps['fq']:.1f}",f"{steps['vcpo']:.1f}"])
b['MATCH_TABLE_NEW']=table('lrrr',r'Model & Matched gap & Step: $F_q$ & Step: $\gC$',rows,r'HumanEval fit matching at 90\% of each run\textquotesingle s common observed interval. Steps are interpolated and averaged; OLMo has one seed. Gaps are in nats/token.','tab:match-new')
rows=[]
for size in ['0.8B','2B']:
 u=json.loads((ROOT/f'tmp/e6_temperature_raw/e6_single_cert_null_Qwen3.5-{size}.json').read_text())
 for k,nm in zip(ks,kn):
  r=u['per_component'][k];rows.append([size,nm,f"{r['cos_single_at_i_star']:.3f}",f"{r['percentile_of_i_star']:.1f}",f"{100*r['selfcheck_rel_err']:.1f}\\%"])
b['NULL_TABLE_NEW']=table('llrrr','Model & Component & Cosine & Pctl. & Discrep.',rows,r'Dominant-reference cosine, percentile among all 32 single-reference cosines, and relative discrepancy from the saved weighted cosine. Percentiles are descriptive.','tab:null-new')
rows=[]
for t,r in tc['by_T'].items():rows.append([t,f"{r['ess']:.2f}",*[f"{r['cos'][k]:.3f}" for k in ks]])
rows.append([r'$\infty$','32.00',*[f"{tc['cos_with_grad_Fq'][k]:.3f}" for k in ks]])
b['TEMPERATURE_TABLE_NEW']=table('rrrrr',r'$T$ & ESS & Teacher CE & Log-shift & Mask-only',rows,r'Gradient temperature measurements on Qwen3.5-0.8B/HumanEval. The infinite-temperature row is the equal-weight reference, not a finite-temperature run.','tab:temperature-new')
b['OBJECTIVE_FIGURE']=r'''\begin{figure}[t]
\centering
\begin{tikzpicture}[font=\small,box/.style={draw,rounded corners=2pt,align=center,text width=3.22cm,minimum height=1.0cm},>=Stealth]
\node[box,fill=blue!5] (f) at (0,0) {Fixed reference source $q$\\$R\,s_\theta$};
\node[box,fill=orange!8] (j) at (3.9,0) {Covered proposal $q$\\$(p_\theta/q)R\,s_\theta$};
\node[align=center] (fl) at (0,-1.5) {$\nabla F_q$\\Reward-weighted likelihood};
\node[align=center] (jl) at (3.9,-1.5) {$\nabla J$\\Expected reward};
\draw[->,thick] (f) -- (fl);
\draw[->,thick] (j) -- (jl);
\end{tikzpicture}
\caption{The objective depends on the sampling source and its weights. The right-hand identity additionally requires coverage of the reward-gradient support. A single supplied reference generally lacks that coverage.}
\label{fig:objectives-new}
\end{figure}'''
plots=[]
for k,nm,color,mark in zip(ks,kn,['blue!70!black','orange!85!black','teal!80!black'],['*','square*','triangle*']):
 coords=' '.join(f"({r['ess']/32:.6f},{r['cos'][k]:.6f})" for r in tc['by_T'].values())
 plots += [rf'\addplot[{color},thick,mark={mark}] coordinates {{{coords}}};',rf'\addlegendentry{{{nm}}}',rf'\addplot[{color},dashed,forget plot,domain=0:1] {{{tc["cos_with_grad_Fq"][k]}}};']
b['TEMPERATURE_FIGURE_NEW']=r'''\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=5.5cm,xmin=0,xmax=1,ymin=0,ymax=.38,
 xlabel={Effective sample fraction (ESS/32)},ylabel={Cosine with component},
 tick label style={font=\footnotesize},label style={font=\small},
 grid=major,grid style={gray!15},legend style={font=\footnotesize,at={(.5,-.29)},anchor=north,legend columns=3,draw=none}]
'''+ '\n'.join(plots)+r'''
\end{axis}
\end{tikzpicture}
\caption{Temperature intervention on Qwen3.5-0.8B/HumanEval. Markers are $T=1,12.002,55.312,64$, joined as a visual guide. Dashed lines are equal-weight component cosines. Higher scalar coverage brings measured similarities toward these likelihood endpoints.}
\label{fig:temperature-new}
\end{figure}'''
rows=[]
lengths=[5,10,15,20,25,30,40,50]
for ds in dsname:
 for m in mods:
  r=get('decay',m,ds)
  rows.append([dsname[ds],names[m],*[(f"{r['horizon'][str(L)]:.3f}" if str(L) in r['horizon'] else '--') for L in lengths]])
b['HORIZON_TABLE_NEW']=table('llrrrrrrrr',r'Data & Model & $L=5$ & 10 & 15 & 20 & 25 & 30 & 40 & 50',rows,r'Mean prefix $\log_{10}p_\theta$ at each token length, over references at least that long. The eligible task population changes with length; -- denotes an unrecorded length.','tab:horizon-new',True)
rows=[]
ts=['1','1.5','2','3','4','6','8','12','16','24','32','64']
for ds in dsname:
 for m in mods:
  r=next(x for x in temp if x['model']==m and x['dataset']==ds)
  rows.append([dsname[ds],short[m],*[f"{r['ess_vs_T'][T]/r['N']:.3f}" for T in ts]])
b['SCALAR_SWEEP_TABLE_NEW']=table('ll'+'r'*len(ts),'Data & Model & '+' & '.join(ts),rows,r'Complete scalar sweep, reported as ESS divided by the positive reference count. Column headings are temperatures $T$. This sweep covers all 18 cells; it contains no gradient measurements beyond those separately reported.','tab:scalar-new',True).replace(r'\tabcolsep}{4pt}',r'\tabcolsep}{2.3pt}').replace(r'\centering\small',r'\centering\footnotesize')
for key in ['MATCH_TABLE_NEW','NULL_TABLE_NEW','TEMPERATURE_TABLE_NEW']:
 b[key]=b[key].replace(r'\begin{table}[t]',r'\begin{table}[H]')
p=ROOT/'paper/vcpo_rewrite.tex';s=p.read_text()
for key,block in b.items():
 start='% BEGIN '+key;end='% END '+key
 if start in s:
  lo=s.index(start);hi=s.index(end,lo)+len(end);s=s[:lo]+start+'\n'+block+'\n'+end+s[hi:]
 else:
  assert '% '+key in s,key
  s=s.replace('% '+key,start+'\n'+block+'\n'+end)
p.write_text(s)
manifest={'sources':[{'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT/'tmp/e6_temperature_raw').glob('*.json'))], 'positive_margins':54,'allfail_groups':sum(r['allfail'] for r in d['allfail']),'total_groups':sum(r['groups'] for r in d['allfail']),'temperature_gradient_cells':1,'null_model_cells':2}
(ROOT/'paper/REWRITE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built',len(b),'fresh blocks in',p)
