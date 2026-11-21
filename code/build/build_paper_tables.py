"""Insert audited table/plot blocks into the paper; does not run experiments."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
d=json.loads((ROOT/'paper/RESULTS_AUDIT.json').read_text())
p=ROOT/'paper/vcpo_aacl.tex'; text=p.read_text()
mods=['Qwen3.5-0.8B','Qwen3.5-2B','Qwen3.5-4B','Qwen3.5-9B','OLMo-2-1124-7B','glm-4-9b']
names=dict(zip(mods,['Qwen 0.8B','Qwen 2B','Qwen 4B','Qwen 9B','OLMo 7B','GLM 9B']))
def get(key,m,ds):return next(r for r in d[key] if r['model']==m and r['dataset']==ds)
def table(spec,head,rows,caption,label,wide=True):
 env='table*' if wide else 'table'
 return '\\begin{'+env+'}[t]\n\\centering\\small\n\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+head+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{'+env+'}'
blocks={}
rows=[]
for m in [mods[4],mods[0],mods[1],mods[5],mods[2],mods[3]]:
 vals=[]
 for ds in ['humaneval','mbpp']:
  r=get('allfail',m,ds); lo,hi=r['wilson'];vals += [str(r['allfail'])+'/120',f"{r['rate']:.3f}",f'[{lo:.3f}, {hi:.3f}]']
 rows.append([names[m],*vals])
blocks['ALLFAIL_TABLE']=table('lrrrrrr',r'& \multicolumn{3}{c}{HumanEval} & \multicolumn{3}{c}{MBPP}\\'+ '\n'+r'Model & All-fail & Rate & Wilson interval & All-fail & Rate & Wilson interval',rows,r'All-fail counts over 40 tasks $\times$ three seeds, with four rollouts per group. Wilson intervals are pooled at 95\%; gradient checks cover one representative group per run (36 groups, 72 norms).','tab:allfail')
rows=[]
for m in mods:
 vals=[]
 for ds in ['humaneval','mbpp']:
  r=get('decay',m,ds); vals += [f"{r['token']['slope']:.3f}",f"{r['char']['slope']:.4f}",f"{r['token']['r2']:.2f}",f"{r['probability_ess']:.2f}"]
 rows.append([names[m],*vals])
# The full DECAY_APPENDIX table contains these fields and is the single
# numerical record for both the main-text summary and the reserved figure.
rows=[]
for ds in ['humaneval','mbpp']:
 for m in mods:
  rs=[r for r in d['training'] if r['model']==m and r['dataset']==ds]
  if not rs:continue
  r=rs[0]; sd=(f" $\\pm$ {r['sd']:.4f}" if r['sd'] is not None else ' (one seed)')
  rows.append(['HE' if ds=='humaneval' else 'MBPP',names[m],f"${r['mean']:+.4f}$"+sd,f"{r['moves']['fq']:.4f}",f"{r['moves']['vcpo']:.4f}",f"{r['movement_ratio']:.2f}"])
blocks['TRAINING_TABLE']=table('lllrrr',r'Data & Model & Gap difference & Move: $F_q$ & Move: $\gC$ & Ratio',rows,r'Step-300 gap differences $\mathrm{gap}_{F_q}-\mathrm{gap}_{\gC}$ (nats per token), mean $\pm$ sample standard deviation over three seeds except OLMo; positive values mean the uncorrected update has the larger train--held-out gap. Movements are step 300 minus step 0 training fit. MBPP rows are descriptive because movement differs and the $\gC$ curves are nonmonotone.','tab:training')
rows=[]
for m in mods:
 vals=[]
 for ds in ['humaneval','mbpp','quixbugs']:
  r=get('mismatch',m,ds);vals += [f"{r['cos_vcpo_clean_vs_corrupted']:.3f}",f"{r['cos_fq_clean_vs_corrupted']:.3f}"]
 rows.append([names[m],*vals])
blocks['MISMATCH_TABLE']=table('lrrrrrr',r'& \multicolumn{2}{c}{HumanEval} & \multicolumn{2}{c}{MBPP} & \multicolumn{2}{c}{QuixBugs}\\'+ '\n'+r'Model & $\gC$ & $\nabla F_q$ & $\gC$ & $\nabla F_q$ & $\gC$ & $\nabla F_q$',rows,r'Cosine of each update with its own mismatched-certificate version (32 tasks, seed 0). Lower values mean greater directional sensitivity. Labels remain fixed after mismatching; no claim of verifier robustness follows.','tab:mismatch')
plots=[]
for m,col,mark in [(mods[0],'blue','*'),(mods[3],'red','square*')]:
 r=get('decay',m,'humaneval'); a=r['token']['slope'];b=r['token']['intercept']
 plots += [rf'\addplot[thick,{col},domain=6:55] {{{a}*x+({b})}};',rf'\addlegendentry{{{names[m]}, whole-text OLS}}',rf'\addplot[only marks,{col},mark={mark}] coordinates {{'+' '.join(f"({L},{v})" for L,v in sorted(r['horizon'].items(),key=lambda x:int(x[0])))+'};',rf'\addlegendentry{{{names[m]}, eligible prefixes}}']
blocks['DECAY_FIGURE']=r'''\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.97\columnwidth,height=5.3cm,xlabel={Certificate length $L$ (tokens)},ylabel={$\log_{10}\pol(\cert\mid h)$},xmin=4,xmax=55,ymin=-18,ymax=0,grid=major,grid style={gray!20},legend style={font=\scriptsize,at={(0.5,-0.30)},anchor=north,draw=none}]
'''+ '\n'.join(plots)+r'''
\end{axis}
\end{tikzpicture}
\caption{Whole-certificate OLS with fitted intercepts and eligible-prefix mean log-probabilities on HumanEval. Lines show only the short-length part of the observed range; markers average over tasks with at least $L$ tokens. They estimate different quantities.}
\label{fig:decay}
\end{figure}'''
rows=[]
for ds in ['humaneval','mbpp','quixbugs']:
 for m in mods:
  r=get('alignment',m,ds);vals=[]
  for k in ['hsd','opid','edge_opd']:
   a=r['alignment'][k]; vals += [f"{a['cos_with_grad_Fq']:.3f}",f"{a['cos_with_grad_VCPO']:.3f}"]
  dlabel={'humaneval':'HE','mbpp':'MBPP','quixbugs':'QuixBugs'}[ds]
  mlabel=names[m]+(r'$^\dagger$' if ds=='humaneval' and m=='glm-4-9b' else '')
  rows.append([dlabel,mlabel,*vals])
blocks['ALIGN_APPENDIX']=table('llrrrrrr',r'& & \multicolumn{2}{c}{Teacher CE} & \multicolumn{2}{c}{Log-shift} & \multicolumn{2}{c}{Mask-only}\\'+ '\n'+r'Data & Model & $\nabla F_q$ & $\gC$ & $\nabla F_q$ & $\gC$ & $\nabla F_q$ & $\gC$',rows,r'Complete diagnostic cosines. HumanEval uses the shared 64-task subset; MBPP and QuixBugs use 16 tasks (seed 0). $\dagger$: the GLM/HumanEval row remains in the 54-cell result and is flagged because both references have near-zero cosine and its mismatched-gradient norm is anomalous.','tab:align')
rows=[]
for ds in ['humaneval','mbpp','quixbugs']:
 for m in mods:
  r=get('decay',m,ds);rows.append([{'humaneval':'HE','mbpp':'MBPP','quixbugs':'QuixBugs'}[ds],names[m],f"{r['token']['intercept']:.3f}",f"{r['token']['slope']:.3f}",f"{r['char']['intercept']:.3f}",f"{r['char']['slope']:.4f}",f"{r['token']['r2']:.2f}",f"{r['probability_ess']:.2f}"])
blocks['DECAY_APPENDIX']=table('llrrrrrr',r'Data & Model & Token intercept & /token & Char intercept & /char & $R^2$ & ESS',rows,'Full saved OLS fits and probability-weighted ESS. QuixBugs contains 40 certificates; HumanEval and MBPP contain 164 and 257.','tab:decay-all')
rows=[]
for r in d['training']:
 if r['dataset']!='humaneval':continue
 steps={a:sum(x['fit_match_90']['steps'][a] for x in r['runs'])/len(r['runs']) for a in ['fq','vcpo']}
 rows.append([names[r['model']],f"${r['match90']:+.4f}$",f"{steps['fq']:.1f}",f"{steps['vcpo']:.1f}",f"{r['moves']['fq']:.4f}",f"{r['moves']['vcpo']:.4f}"])
blocks['MATCH_TABLE']=table('lrrrrr',r'Model & Matched gap difference & Step: $F_q$ & Step: $\gC$ & Move: $F_q$ & Move: $\gC$',rows,'HumanEval matching at 90\% of each run\'s common observed fit range. Gap difference is $\mathrm{gap}_{F_q}-\mathrm{gap}_{\gC}$; positive values mean the uncorrected update has the larger gap. Steps are interpolated and averaged across seeds (one seed for OLMo).','tab:matched')
for k,v in blocks.items():
 block=f'% BEGIN {k}\n{v}\n% END {k}'
 pattern=rf'% BEGIN {k}\n.*?% END {k}'
 if re.search(pattern,text,re.S):text=re.sub(pattern,lambda _:block,text,flags=re.S)
 else:
  assert '% '+k in text,k
  text=text.replace('% '+k,block)
p.write_text(text)
print('Inserted',len(blocks),'audited blocks')
