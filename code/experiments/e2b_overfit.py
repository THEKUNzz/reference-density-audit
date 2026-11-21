#!/usr/bin/env python3
"""E2(b): does the uncorrected (no-ratio / F_q) update overfit like SFT?

Train a few steps with the F_q update (reward-weighted log-likelihood on the
certificates) and record two curves:
  - train fit: mean log-prob on the TRAIN certificates (R=1 reference solutions)
  - held-out fit: mean log-prob on the HELD-OUT certificates

An SFT-like objective fits the train distribution fast and plateaus or drops
on held-out (overfitting). A genuine RL objective (grad J) would not show this
signature. We compare the no-ratio update against the corrected (VCPO) update.

**F_q must not degenerate to SFT.** The train batch therefore mixes
  - R=1 items: the canonical reference solutions (the "certificates"), and
  - R=0 items: on-policy sampled rollouts that fail the unit test.
``train_step_fq`` weights each item by its reward (F_q = (1/n) Σ R_i log p_θ),
so with non-constant R it is genuinely reward-weighted and not SFT. The fit
curves are still measured only on the R=1 certificates.

Performance (correctness-neutral): train/held-out encodings are computed once
and reused; forwards run in chunks of ``CHUNK`` sequences to bound the
[chunk, Lmax, vocab] logits tensor; the fit is recorded every 10 steps.
"""
import argparse
import json
import torch
import numpy as np
from dataclasses import asdict

import vcpo_common as common
from vcpo_common import load_model, encode_with_boundary, write_result, RunMeta
from e1_formal import sample_rollouts, run_humaneval_test, MBPP_STOP, STOP_SEQUENCES

CHUNK = 4  # sequences per forward; bounds the [chunk, Lmax, vocab] logits tensor


def encode_batch(tokenizer, items):
    """Right-pad ``items`` (list of (prompt, completion)) into one tensor.

    Returns ``(ids [B, Lmax], starts, lens, mask [B, Lmax])``.
    """
    rows, starts, lens = [], [], []
    for prompt, cert in items:
        ids, start = encode_with_boundary(tokenizer, prompt, cert)
        rows.append(ids[0])
        starts.append(start)
        lens.append(int(ids.shape[1]))
    Lmax = max(lens)
    B = len(items)
    pad_id = tokenizer.pad_token_id
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.long)
    for b, (row, L) in enumerate(zip(rows, lens)):
        ids[b, :L] = row
        mask[b, :L] = 1
    return ids, starts, lens, mask


def batch_token_logprobs(model, ids, starts, lens, mask, device, chunk=CHUNK):
    """Per-token log-probs of each completion span, chunked batched forwards."""
    out = []
    B = len(starts)
    for b0 in range(0, B, chunk):
        b1 = min(b0 + chunk, B)
        ids_c = ids[b0:b1].to(device)
        mask_c = mask[b0:b1].to(device)
        logits = model(ids_c, attention_mask=mask_c).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        for j in range(b1 - b0):
            b = b0 + j
            s = starts[b]
            L = lens[b]
            lp = log_probs[j, s - 1 : L - 1]
            tgt = ids_c[j, s:L]
            out.append(lp.gather(1, tgt.unsqueeze(1)).squeeze(1))
        del logits, log_probs
    return out


@torch.no_grad()
def mean_logprob(model, enc, device):
    """Mean per-token log-prob over a pre-encoded batch."""
    ids, starts, lens, mask = enc
    per_token = batch_token_logprobs(model, ids, starts, lens, mask, device)
    means = [float(pt.sum()) / (lens[b] - starts[b]) for b, pt in enumerate(per_token)]
    return float(np.mean(means))


def _sgd_step(model, lr):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
    model.zero_grad(set_to_none=True)


def _flat_grad_of(model, loss_fn):
    """Compute the flat (bf16 CPU) gradient of ``loss_fn()``."""
    model.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    g = common.flat_grad(model)
    model.zero_grad(set_to_none=True)
    return g


def diagnostic_cos_fq_sft(model, ids, starts, lens, mask, rewards, device):
    """cos(grad F_q, grad SFT) on the train batch.

    F_q = (1/n) Σ R_i log p_θ(τ_i) (reward-weighted); SFT = (1/n) Σ log p_θ(τ_i)
    (uniform). With non-constant R the two gradients must differ; a cosine
    close to 1 would mean the reward weighting did not actually make the test
    non-trivial. This is the direct evidence the fix took effect.
    """
    def loss_fq():
        per_token = batch_token_logprobs(model, ids, starts, lens, mask, device)
        return -sum(r * pt.sum() for r, pt in zip(rewards, per_token)) / len(starts)

    def loss_sft():
        per_token = batch_token_logprobs(model, ids, starts, lens, mask, device)
        return -sum(pt.sum() for pt in per_token) / len(starts)

    g_fq = _flat_grad_of(model, loss_fq)
    g_sft = _flat_grad_of(model, loss_sft)
    cos = common.cosine(g_fq, g_sft)
    del g_fq, g_sft
    torch.cuda.empty_cache()
    return float(cos)


def _chunked_backward(model, ids, starts, lens, mask, weights, device, normalize, lr=5e-4, chunk=CHUNK):
    """Forward+backward per chunk, accumulating gradients.

    A single backward over the whole ~143-item batch retains every item's
    [Lmax, vocab] float32 logits/log-probs in the autograd graph (~50 GB) and
    OOMs. Backward per chunk (and free that chunk's graph) bounds memory to one
    chunk. ``weights[i]`` is the scalar coefficient of item i (0 for R=0);
    ``normalize`` divides the loss by n (F_q does, VCPO's max-normalized
    coefficients do not).
    """
    n = len(starts)
    model.zero_grad(set_to_none=True)
    for b0 in range(0, n, chunk):
        b1 = min(b0 + chunk, n)
        ids_c = ids[b0:b1].to(device)
        mask_c = mask[b0:b1].to(device)
        logits = model(ids_c, attention_mask=mask_c).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        chunk_total = None
        for j in range(b1 - b0):
            b = b0 + j
            w = weights[b]
            if w == 0.0:
                continue
            s = starts[b]
            L = lens[b]
            lp = log_probs[j, s - 1 : L - 1]
            tgt = ids_c[j, s:L]
            contrib = w * lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum()
            chunk_total = contrib if chunk_total is None else chunk_total + contrib
        # A chunk of all R=0 items contributes zero; skip its backward (its
        # chunk_total is None and a leaf 0-tensor has no grad_fn to backward).
        if chunk_total is not None:
            loss = -chunk_total / n if normalize else -chunk_total
            loss.backward()
            del loss
        del logits, log_probs
    _sgd_step(model, lr)


def train_step_fq(model, ids, starts, lens, mask, rewards, device, lr=5e-4):
    """One SGD step with F_q = (1/n) Σ R_i · log p_θ(τ_i).

    Reward-weighted, NOT the unweighted SFT loss. With non-constant R this
    differs from SFT; R=0 items contribute zero to the numerator but still
    count in the proposal size n.
    """
    _chunked_backward(model, ids, starts, lens, mask, rewards, device, normalize=True, lr=lr)


def train_step_vcpo(model, ids, starts, lens, mask, rewards, device, lr=5e-4):
    """One SGD step with the VCPO (corrected) update, coefficients detached.

    The certificate coefficient c = p_θ/(n·p_θ + q) applies only to R=1 items;
    R=0 items have reward 0 so they contribute nothing. Coefficients are
    computed in log space and normalized by the batch max (scale-invariant).
    """
    with torch.no_grad():
        detached = batch_token_logprobs(model, ids, starts, lens, mask, device)
        log_ps = [float(pt.sum()) for pt in detached]
    n = len(starts)
    log_cs = []
    for r, log_p in zip(rewards, log_ps):
        if r > 0.0:
            log_D = float(np.logaddexp(np.log(n) + log_p, 0.0))
            log_cs.append(log_p - log_D)
        else:
            log_cs.append(None)  # R=0 -> coefficient 0
    max_log_c = max(lc for lc in log_cs if lc is not None)

    weights = [0.0 if lc is None else float(np.exp(lc - max_log_c)) for lc in log_cs]
    _chunked_backward(model, ids, starts, lens, mask, weights, device, normalize=False, lr=lr)


def build_train_with_rewards(model, tokenizer, rows, device, n_train=40, n_rollouts=4, dataset="humaneval"):
    """Train batch: R=1 canonical solutions + R=0 on-policy failures.

    Returns a list of (prompt, completion, reward). For each task we keep the
    canonical solution (R=1) and every on-policy rollout that fails the test
    (R=0), so R is non-constant and F_q != SFT.
    """
    stops = MBPP_STOP if dataset == "mbpp" else STOP_SEQUENCES
    train = []
    for i, r in enumerate(rows[:n_train]):
        train.append((r["prompt"], r["certificate"], 1.0))
        rollouts = sample_rollouts(model, tokenizer, r["prompt"], device, n=n_rollouts, stop_sequences=stops)
        for rollout in rollouts:
            if dataset == "mbpp":
                passed = common.run_mbpp_test(rollout, r["test_list"], r["test_imports"])
            else:
                passed = run_humaneval_test(r["prompt"], rollout, r["test"], r["entry_point"])
            if not passed:
                train.append((r["prompt"], rollout, 0.0))
        # generate() with the linear-attention fallback leaks ~0.5 GB per rollout;
        # free it each task or 160 generations accumulate to ~78 GB and OOM.
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-heldout", type=int, default=40)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n-rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="humaneval", choices=sorted(common.LOADERS))
    parser.add_argument("--outdir", default=None,
                        help="results subdir (e.g. 'kernel'); None = top-level results/")
    args = parser.parse_args()

    common.set_seed(args.seed)
    # Load in EVAL mode for the R=0 on-policy sampling: train=True enables
    # gradient checkpointing which corrupts generate() (garbage rollouts). The
    # training loop below reloads the model with train=True.
    model, tokenizer = load_model(args.model, train=False)
    device = next(model.parameters()).device

    rows = common.load_dataset_items(args.dataset)
    train_mixed = build_train_with_rewards(model, tokenizer, rows, device,
                                           n_train=args.n_train, n_rollouts=args.n_rollouts,
                                           dataset=args.dataset)
    n_r1 = sum(1 for _, _, r in train_mixed if r > 0)
    n_r0 = sum(1 for _, _, r in train_mixed if r == 0)
    print(f"train batch: n_R1={n_r1} n_R0={n_r0} (R non-constant: {n_r0 > 0})", flush=True)

    # fit is measured only on the R=1 certificates
    train_certs = [(r["prompt"], r["certificate"]) for r in rows[: args.n_train]]
    heldout_certs = [(r["prompt"], r["certificate"]) for r in rows[args.n_train : args.n_train + args.n_heldout]]

    train_enc = encode_batch(tokenizer, [it[:2] for it in train_mixed])
    rewards = [it[2] for it in train_mixed]
    train_cert_enc = encode_batch(tokenizer, train_certs)
    heldout_enc = encode_batch(tokenizer, heldout_certs)

    # Sampling is done; free the eval-mode model before the training reloads.
    del model
    torch.cuda.empty_cache()

    # Diagnostic on a SEPARATE train-mode model (initial weights, same scope as
    # the smoke). Keeping it off the training model matters: flat_grad's
    # backward + detach + zero_grad corrupts the model for a later forward, so
    # the subsequent train_step's backward sees "does not require grad".
    DIAG_N = 16
    model_diag, _ = load_model(args.model, train=True)
    diag_device = next(model_diag.parameters()).device
    cos_fq_sft = diagnostic_cos_fq_sft(
        model_diag,
        train_enc[0][:DIAG_N], train_enc[1][:DIAG_N], train_enc[2][:DIAG_N], train_enc[3][:DIAG_N],
        rewards[:DIAG_N], diag_device)
    print(f"  cos(grad F_q, grad SFT) = {cos_fq_sft:.4f}  (want clearly < 1)", flush=True)
    del model_diag
    torch.cuda.empty_cache()

    curves = {}
    for name, step_fn in (("fq", train_step_fq), ("vcpo", train_step_vcpo)):
        model, tokenizer = load_model(args.model, train=True)
        model.requires_grad_(True)
        device = next(model.parameters()).device
        series = []
        for s in range(args.steps + 1):
            if s > 0:
                step_fn(model, *train_enc, rewards, device)
            if s % 10 == 0:
                tr = mean_logprob(model, train_cert_enc, device)
                ho = mean_logprob(model, heldout_enc, device)
                series.append({"step": s, "train_fit": tr, "heldout_fit": ho})
                print(f"  {name} step={s}: train={tr:.4f} heldout={ho:.4f}", flush=True)
        curves[name] = series

    gap_fq = curves["fq"][-1]["train_fit"] - curves["fq"][-1]["heldout_fit"]
    gap_vcpo = curves["vcpo"][-1]["train_fit"] - curves["vcpo"][-1]["heldout_fit"]

    meta = RunMeta(
        experiment="e2b_overfit",
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        dtype="bfloat16",
        extra={"n_train": args.n_train, "n_heldout": args.n_heldout, "steps": args.steps,
               "n_R1": n_r1, "n_R0": n_r0, "n_rollouts": args.n_rollouts,
               "metric": "mean per-token log-prob on R=1 certificates; gap = train - heldout",
               "batched": True, "chunk": CHUNK, "reward_weighted": True,
               "cos_fq_vs_sft": cos_fq_sft},
    ).fill_environment()
    payload = {"curves": curves, "gap_fq": gap_fq, "gap_vcpo": gap_vcpo,
               "n_R1": n_r1, "n_R0": n_r0, "cos_fq_vs_sft": cos_fq_sft}
    dsuffix = "" if args.dataset == "humaneval" else f"_{args.dataset}"
    outdir = common.RESULTS_DIR if args.outdir is None else common.RESULTS_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"e2b_{args.model.split('/')[-1]}_seed{args.seed}{dsuffix}.json"
    with path.open("w") as handle:
        json.dump({"meta": asdict(meta), **payload}, handle, indent=2, sort_keys=True)

    print(f"model={args.model}")
    print(f"  F_q   (reward-weighted) gap train-heldout = {gap_fq:+.4f}")
    print(f"  VCPO  (corrected)      gap train-heldout = {gap_vcpo:+.4f}")
    print(f"  (larger gap = more SFT-like overfitting)")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
