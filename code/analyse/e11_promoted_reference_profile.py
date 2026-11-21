#!/usr/bin/env python3
"""E11: what does the sequence-probability weight actually select for?

Reconstructs each diagnostic batch exactly as e6_ess_all.py builds it
(humaneval = random.Random(0).sample(items, 64); mbpp/quixbugs = items[:16];
positives are the even indices) and asks where the promoted certificate
i* sits among its batch on three orderings: sequence log-probability
(rank 1 by construction), per-token fit log p / L, and token length.

Then, over each full reference pool, it reports the Spearman correlation
of sequence log-probability with per-token fit and with shortness.

Input:  results/e3_<model>_<dataset>.json   (per-reference teacher-forced logprobs)
Check:  the reconstructed ESS must reproduce results/tempering_reconstructed.json
Output: results/promoted_reference_profile.json
"""
import json, os, random
import numpy as np

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_ROOT)   # paths below are repo-root relative


MODELS = ['Qwen3.5-0.8B', 'Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B',
          'OLMo-2-1124-7B', 'glm-4-9b']
DATASETS = {'humaneval': (64, 0), 'mbpp': (16, None), 'quixbugs': (16, None)}


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v))
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    a, b = rank(x), rank(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def rank_of(vals, i, descending=True):
    v = np.asarray(vals, float)
    order = np.argsort(-v if descending else v)
    return int(np.where(order == i)[0][0]) + 1


def e3_path(model, ds):
    """Qwen records sit at the top level; OLMo and GLM under results/verified/.
    results/verified/ is canonical: it reproduces the archived first-run ESS
    exactly, whereas results/server2/ disagrees on MBPP by up to 6.3 nats."""
    for cand in (f'data/results/e3_{model}_{ds}.json',
                 f'data/results/verified/e3_{model}_{ds}.json'):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f'no e3 record for {model}/{ds}')


def batch_indices(n, n_tasks, shuffle_seed):
    idx = (random.Random(shuffle_seed).sample(range(n), min(n_tasks, n))
           if shuffle_seed is not None else list(range(min(n_tasks, n))))
    return [idx[i] for i in range(0, len(idx), 2)]   # even indices carry R=1


def main():
    try:
        archived = json.load(open('data/results/tempering_reconstructed.json'))
    except FileNotFoundError:
        archived = {}
    out = []
    for model in MODELS:
        for ds, (n_tasks, seed) in DATASETS.items():
            pts = json.load(open(e3_path(model, ds)))['points']
            pos = batch_indices(len(pts), n_tasks, seed)
            logp = np.array([pts[i]['logprob_sum'] for i in pos])
            fit = np.array([pts[i]['logprob_mean_per_token'] for i in pos])
            length = np.array([pts[i]['n_tokens'] for i in pos], float)

            log_c = logp - np.logaddexp(np.log(n_tasks) + logp, 0.0)
            a = np.exp(log_c - log_c.max())
            ess = float((a.sum() ** 2) / (a ** 2).sum())
            istar = int(np.argmax(log_c))

            ref = archived.get(f'{model}/{ds}', {}).get('ess')
            row = {
                'model': model, 'dataset': ds, 'n_positive': len(pos),
                'ess_reconstructed': ess, 'ess_archived_first_run': ref,
                'ess_matches_archive': (ref is not None and abs(ess - ref) < 1e-3),
                'i_star': istar,
                'rank_by_sequence_logprob': rank_of(logp, istar, True),
                'rank_by_per_token_fit': rank_of(fit, istar, True),
                'rank_by_shortness': rank_of(length, istar, False),
                'i_star_per_token_fit': float(fit[istar]),
                'best_per_token_fit': float(fit.max()),
                'i_star_tokens': int(length[istar]),
                'median_tokens': float(np.median(length)),
                # full-pool orderings
                'pool_n': len(pts),
                'rho_seqlogp_per_token_fit': spearman(
                    [p['logprob_sum'] for p in pts],
                    [p['logprob_mean_per_token'] for p in pts]),
                'rho_seqlogp_shortness': spearman(
                    [p['logprob_sum'] for p in pts],
                    [-p['n_tokens'] for p in pts]),
            }
            out.append(row)
            print(json.dumps(row), flush=True)

    bad = [r for r in out if r['ess_archived_first_run'] is not None
           and not r['ess_matches_archive']]
    print(f"\nreconstruction check: {len(out) - len(bad)}/{len(out)} cells reproduce the archived ESS")
    for r in bad:
        print(f"  MISMATCH {r['model']}/{r['dataset']}: "
              f"{r['ess_reconstructed']:.6f} vs {r['ess_archived_first_run']:.6f}")
    with open('data/results/promoted_reference_profile.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("WROTE data/results/promoted_reference_profile.json")


if __name__ == '__main__':
    main()
