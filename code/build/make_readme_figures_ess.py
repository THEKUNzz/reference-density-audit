import json, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
SURF = '#fcfcfb'

rows = json.load(open('data/results/promoted_reference_profile.json'))
order = {'humaneval': 'HumanEval', 'mbpp': 'MBPP', 'quixbugs': 'QuixBugs'}
by = collections.defaultdict(list)
for r in rows:
    by[r['dataset']].append(r)

labels, ess, pool, seps = [], [], [], []
y = 0
ticks = []
for ds, name in order.items():
    cells = sorted(by[ds], key=lambda x: x['ess_reconstructed'])
    for c in cells:
        labels.append(c['model'].replace('Qwen3.5-', 'Qwen ').replace('OLMo-2-1124-', 'OLMo ').replace('glm-4-9b', 'GLM 9B'))
        ess.append(c['ess_reconstructed']); pool.append(c['n_positive']); ticks.append(y); y += 1
    seps.append(y - 0.5); y += 0.8

fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

for t, e, n in zip(ticks, ess, pool):
    ax.plot([e, n], [t, t], color=GRID, lw=2, solid_capstyle='round', zorder=1)
    ax.plot(n, t, 'o', ms=7, mfc=SURF, mec=ORANGE, mew=2, zorder=3)
    ax.plot(e, t, 'o', ms=8, color=BLUE, mec=SURF, mew=1.5, zorder=4)

ax.set_xscale('log')
ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9, color=MUTED)
ax.invert_yaxis()
ax.set_xlim(0.85, 46)
for s in seps[:-1]:
    ax.axhline(s, color=GRID, lw=0.8)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(axis='x', colors=MUTED, labelsize=9, length=3, color=GRID)
ax.tick_params(axis='y', length=0)
ax.xaxis.grid(True, color=GRID, lw=0.6, alpha=0.6); ax.set_axisbelow(True)

bands = [(0, 'HumanEval  ·  32 references'), (6, 'MBPP  ·  8 references'), (12, 'QuixBugs  ·  8 references')]
for lo, name in bands:
    ax.text(0.0, ticks[lo] - 0.95, name, transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=9.5, color=INK, fontweight='bold')

ax.set_xlabel('effective sample size  (log scale)', fontsize=10, color=MUTED, labelpad=8)
ax.set_title('The weight keeps about one reference, whatever the pool size',
             fontsize=12.5, color=INK, fontweight='bold', loc='left', pad=14)
ax.legend(handles=[
    Line2D([], [], marker='o', ls='', ms=8, color=BLUE, mec=SURF, mew=1.5, label='measured effective sample size'),
    Line2D([], [], marker='o', ls='', ms=7, mfc=SURF, mec=ORANGE, mew=2, label='references supplied (no collapse)'),
], loc='lower right', frameon=False, fontsize=9, labelcolor=MUTED, handletextpad=0.6)

fig.tight_layout()
fig.savefig('assets/ess-collapse.png', facecolor=SURF, bbox_inches='tight')
print('wrote assets/ess-collapse.png')
