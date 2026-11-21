"""Verify chunked gradient accumulation == full-batch backward (cos ~1).

Backward only (no SGD step) so the accumulated gradient is still present to
extract via common.flat_grad. lr is irrelevant here because we never update.
"""
import os
import sys
sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import torch
import vcpo_common as common
from vcpo_common import load_model
import e2b_overfit as E

model, tok = load_model("Qwen/Qwen3.5-0.8B", train=True)
model.requires_grad_(True)
dev = next(model.parameters()).device

from datasets import load_dataset
rows = list(load_dataset("openai/openai_humaneval", split="test"))[:8]
items = [(r["prompt"], r["canonical_solution"]) for r in rows]
rewards = [1.0, 0.0] * 4
ids, starts, lens, mask = E.encode_batch(tok, items)
n = len(rewards)


def _backward(weights, normalize, chunk):
    model.zero_grad(set_to_none=True)
    for b0 in range(0, n, chunk):
        b1 = min(b0 + chunk, n)
        ids_c = ids[b0:b1].to(dev)
        mask_c = mask[b0:b1].to(dev)
        logits = model(ids_c, attention_mask=mask_c).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        ct = None
        for j in range(b1 - b0):
            b = b0 + j
            w = weights[b]
            if w == 0.0:
                continue
            s = starts[b]; L = lens[b]
            lp = log_probs[j, s - 1 : L - 1]
            tgt = ids_c[j, s:L]
            contrib = w * lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum()
            ct = contrib if ct is None else ct + contrib
        if ct is not None:
            loss = -ct / n if normalize else -ct
            loss.backward()
            del loss
        del logits, log_probs
    return common.flat_grad(model)


# VCPO coefficients need the global max, computed the same way as train_step_vcpo.
with torch.no_grad():
    detached = E.batch_token_logprobs(model, ids, starts, lens, mask, dev)
    log_ps = [float(pt.sum()) for pt in detached]
log_cs = []
for r, log_p in zip(rewards, log_ps):
    if r > 0.0:
        log_D = float(__import__("numpy").logaddexp(__import__("numpy").log(n) + log_p, 0.0))
        log_cs.append(log_p - log_D)
    else:
        log_cs.append(None)
max_log_c = max(lc for lc in log_cs if lc is not None)
vcpo_weights = [0.0 if lc is None else float(__import__("numpy").exp(lc - max_log_c)) for lc in log_cs]

for name, weights, normalize in (("fq", rewards, True), ("vcpo", vcpo_weights, False)):
    g_chunked = _backward(weights, normalize, chunk=4)
    g_full = _backward(weights, normalize, chunk=n)
    cos = float(torch.nn.functional.cosine_similarity(g_chunked, g_full, dim=0))
    ratio = float(g_chunked.norm() / g_full.norm()) if g_full.norm() > 0 else float("nan")
    ok = "OK" if cos > 0.999 and abs(ratio - 1) < 0.01 else "MISMATCH"
    print(f"{name:5s}  cos={cos:.6f}  norm_ratio={ratio:.4f}  {ok}")
