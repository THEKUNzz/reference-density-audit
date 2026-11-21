import json, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#d9d8d4', '#fcfcfb'

rows = json.load(open('data/results/promoted_reference_profile.json'))
by = collections.defaultdict(list)
for r in rows: by[r['dataset']].append(r)
order = [('humaneval','HumanEval'), ('mbpp','MBPP'), ('quixbugs','QuixBugs\n(lengths near-uniform)')]

fig, ax = plt.subplots(figsize=(8.6, 2.9), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

ypos = []
for i,(ds,name) in enumerate(order):
    cells = by[ds]
    short = [c['rho_seqlogp_shortness'] for c in cells]
    fit   = [c['rho_seqlogp_per_token_fit'] for c in cells]
    y = len(order)-1-i; ypos.append(y)
    ax.plot([min(fit), max(short)], [y, y], color=GRID, lw=2, solid_capstyle='round', zorder=1)
    ax.plot(short, [y]*len(short), 'o', ms=8, color=ORANGE, mec=SURF, mew=1.2, alpha=.9, zorder=3)
    ax.plot(fit,   [y]*len(fit),   'o', ms=8, color=BLUE,   mec=SURF, mew=1.2, alpha=.9, zorder=3)

ax.set_yticks(ypos); ax.set_yticklabels([n for _,n in order], fontsize=10, color=INK)
ax.set_ylim(-0.6, len(order)-0.4)
ax.set_xlim(0, 1.0); ax.set_xticks([0,.2,.4,.6,.8,1.0])
for sp in ('top','right','left'): ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(axis='x', colors=MUTED, labelsize=9, length=3, color=GRID)
ax.tick_params(axis='y', length=0)
ax.xaxis.grid(True, color=GRID, lw=.6, alpha=.6); ax.set_axisbelow(True)
ax.set_xlabel('Spearman correlation with sequence log-probability', fontsize=10, color=MUTED, labelpad=8)
ax.set_title('What the weight ranks by: brevity, not fit',
             fontsize=12.5, color=INK, fontweight='bold', loc='left', pad=12)
ax.legend(handles=[
    Line2D([],[],marker='o',ls='',ms=8,color=ORANGE,mec=SURF,mew=1.2,label='shortness'),
    Line2D([],[],marker='o',ls='',ms=8,color=BLUE,  mec=SURF,mew=1.2,label='per-token fit'),
], loc='upper center', ncol=2, frameon=False, fontsize=9.5, labelcolor=MUTED,
   bbox_to_anchor=(0.5,-0.34), handletextpad=0.5, columnspacing=2.2)
fig.tight_layout()
fig.savefig('assets/brevity-not-fit.png', facecolor=SURF, bbox_inches='tight')
print('wrote assets/brevity-not-fit.png')
