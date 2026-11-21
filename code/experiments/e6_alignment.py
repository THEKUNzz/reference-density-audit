#!/usr/bin/env python3
"""E6: do published privileged-distillation updates align with grad F_q?

This is the direct test of the paper's central claim. Section 3 argues that a
broad class of privileged updates can be read as proposal substitution without
the density ratio, and that such an update is the exact gradient of

    F_q(theta) = \\int q(tau) R(tau) log p_theta(tau) dmu

rather than a poor approximation of grad J. Until now that claim has only been
tested against our own "VCPO without ratio" construction, never against an
actual published method.

Four update rules are implemented from the formulas in their papers, evaluated
on one shared batch:

  HSD       per-token reverse KL to a teacher conditioned on a successful peer
            (teacher stop-gradient)
  OPID      log-prob difference between skill-augmented and plain contexts,
            used as a token-level advantage
  EDGE-OPD  hard mask 1[e_t > 0] on an on-policy distillation loss, where
            e_t is the teacher's log-ratio with and without privileged context
  VCPO      the proposal-corrected estimator (ours), as a control

IMPORTANT SCOPE LIMIT: these are the *update rules*, not the authors' full
systems -- no skill mining, no routing, no hyperparameter search. The paper may
therefore say only that "the update implied by their formula aligns with grad
F_q at the gradient level", never that we reproduced their method.

A low alignment for any method is a real result: it would mean the diagnosis
does not cover that method, and the paper's scope must shrink accordingly.
Do not tune to make the numbers come out.

Usage:
    python e6_alignment.py --model Qwen/Qwen3-0.6B-Base --n-tasks 16
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import vcpo_common as common  # noqa: F401
from vcpo_common import RunMeta, cosine, flat_grad, load_model, write_result


def token_logprobs(model, input_ids, start: int) -> torch.Tensor:
    """Per-token log-probs of ``input_ids[start:]`` under teacher forcing."""

    logits = model(input_ids).logits.float()
    log_probs = torch.log_softmax(logits[0, start - 1 : -1], dim=-1)
    targets = input_ids[0, start:]
    return log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)


def full_log_softmax(model, input_ids, start: int) -> torch.Tensor:
    """Full-vocabulary log-probs over the completion span."""

    logits = model(input_ids).logits.float()
    return torch.log_softmax(logits[0, start - 1 : -1], dim=-1)


def grad_of(model, loss: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradient = flat_grad(model)
    model.zero_grad(set_to_none=True)  # free per-parameter grads; gradient is already a copy
    return gradient


def build_batch(model, tokenizer, n_tasks: int, device, dataset: str = "humaneval",
                shuffle_seed: int | None = None) -> list[dict]:
    """One shared batch of (prompt, rollout, certificate, reward) tuples.

    The rollouts are deliberately a mix of R=1 and R=0. That matters: if every
    path in the proposal carried R=1, then F_q would literally *be* the SFT
    loss and the alignment test would be a tautology. The failures are what
    make R non-constant and the test non-trivial.

    R=0 rollouts are REAL on-policy failures: a greedy completion of the prompt
    under the current policy (a base model almost never reproduces the
    reference, so this is a genuine failed attempt rather than a truncated
    reference). R=1 rollouts are the reference certificate.

    ``shuffle_seed`` (when set) draws a random n-task subset instead of the
    task-id prefix, for the small-sample / non-random-selection robustness check.
    """

    import random

    items = common.load_dataset_items(dataset)
    if shuffle_seed is not None:
        items = random.Random(shuffle_seed).sample(items, min(n_tasks, len(items)))
    else:
        items = items[:n_tasks]
    batch = []
    for index, item in enumerate(items):
        certificate = item["certificate"]
        prompt = item["prompt"]
        reward = 1.0 if index % 2 == 0 else 0.0
        rollout = certificate
        if reward == 0.0:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=120, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            rollout = tokenizer.decode(
                generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            if rollout.strip() == certificate.strip():
                reward = 1.0  # greedy reproduced the reference: a success
        batch.append(
            {
                "task_id": item["task_id"],
                "prompt": prompt,
                "rollout": rollout,
                "certificate": certificate,
                "reward": reward,
            }
        )
    return batch


def grad_F_q(model, tokenizer, batch, device) -> torch.Tensor:
    """Analytic grad F_q under the empirical proposal.

    F_q = sum_tau q(tau) R(tau) log p_theta(tau). With q the empirical
    distribution over the batch, this is the reward-weighted mean of the
    sequence log-probabilities. This is the reference quantity, and it is *not*
    the SFT loss unless R happens to be constant.
    """

    total = 0.0
    for item in batch:
        if item["reward"] == 0.0:
            continue  # contributes zero to F_q, but is still in the proposal
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["rollout"])
        total = total + item["reward"] * token_logprobs(model, ids.to(device), start).sum()
    loss = -total / len(batch)
    return grad_of(model, loss)


def grad_F_q_shuffled(model, tokenizer, batch, device, seed=0):
    """Control: F_q on a deranged pairing of prompts and rollouts.

    Same computation, same batch size, same number of R=1 terms -- only the
    prompt-to-certificate correspondence is broken (for R=1 the rollout *is* the
    certificate). Following Du et al. (2018), the control differs from the real
    measurement along exactly one semantic axis, so a gap between them is
    attributable to that axis. A random direction would not serve: in this many
    dimensions its cosine is ~1/sqrt(d) by construction and tests nothing.
    """
    import random

    rng = random.Random(seed)
    rollouts = [item["rollout"] for item in batch]
    for _ in range(100):
        rng.shuffle(rollouts)
        if all(r != item["rollout"] for r, item in zip(rollouts, batch)):
            break
    shuffled = [dict(item, rollout=r) for item, r in zip(batch, rollouts)]
    return grad_F_q(model, tokenizer, shuffled, device)


def grad_sft(model, tokenizer, batch, device) -> torch.Tensor:
    """Plain SFT on the certificates: every certificate, unweighted."""

    total = 0.0
    for item in batch:
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        total = total + token_logprobs(model, ids.to(device), start).sum()
    loss = -total / len(batch)
    return grad_of(model, loss)


def grad_hsd(model, tokenizer, batch, device) -> torch.Tensor:
    """HSD: per-token reverse KL from a reference-conditioned teacher to the student.

    The teacher sees the item's own correct solution as privileged reference
    (for a successful sibling of the same prompt the correct content is exactly
    the certificate). The teacher is stop-gradiented so gradients reach theta
    only through the student. Length normalized per rollout, as in the paper.
    """

    total = 0.0
    used = 0
    for item in batch:
        teacher_prompt = f"{item['prompt']}# reference:\n{item['certificate']}\n"
        student_ids, student_start = common.encode_with_boundary(tokenizer, item["prompt"], item["rollout"])
        teacher_ids, teacher_start = common.encode_with_boundary(tokenizer, teacher_prompt, item["rollout"])

        student = full_log_softmax(model, student_ids.to(device), student_start)
        with torch.no_grad():
            teacher = full_log_softmax(model, teacher_ids.to(device), teacher_start)
        span = min(student.shape[0], teacher.shape[0])
        if span == 0:
            continue
        # KL(teacher || student), the direction used in the paper.
        kl = (teacher[:span].exp() * (teacher[:span] - student[:span])).sum(-1)
        total = total + kl.mean()
        used += 1
    if used == 0:
        raise RuntimeError("HSD needs at least one successful peer in the batch")
    return grad_of(model, total / used)


def grad_opid(model, tokenizer, batch, device) -> torch.Tensor:
    """OPID: skill-augmented minus plain log-probs as a token-level advantage.

    The advantage is detached (computed under the old policy in the paper, here
    under a no-grad pass) and multiplies the student's log-probs. We omit the
    PPO clipping, which at theta = theta_old contributes ratio 1; this is noted
    in the paper's scope limits.
    """

    total = 0.0
    for item in batch:
        skill_prompt = f"{item['prompt']}# hint:\n{item['certificate']}\n"
        plain_ids, plain_start = common.encode_with_boundary(tokenizer, item["prompt"], item["rollout"])
        skill_ids, skill_start = common.encode_with_boundary(tokenizer, skill_prompt, item["rollout"])

        with torch.no_grad():
            plain_old = token_logprobs(model, plain_ids.to(device), plain_start)
            skill_old = token_logprobs(model, skill_ids.to(device), skill_start)
            span = min(plain_old.shape[0], skill_old.shape[0])
            advantage = (skill_old[:span] - plain_old[:span]).detach()
        student = token_logprobs(model, plain_ids.to(device), plain_start)[:span]
        total = total + (advantage * student).sum()
    return grad_of(model, -total / len(batch))


def grad_edge_opd(model, tokenizer, batch, device) -> torch.Tensor:
    """EDGE-OPD: hard evidence mask 1[e_t > 0] restricting the loss support.

    e_t = log pi_T(y_t | x, r, y_<t) - log pi_T(y_t | x, y_<t), threshold 0 as
    in the paper. The mask is stop-gradient: it selects which positions
    contribute, and the authors are explicit that this changes the support of
    the objective rather than reweighting it.
    """

    total = 0.0
    kept = 0
    seen = 0
    for item in batch:
        privileged_prompt = f"{item['prompt']}# reference:\n{item['certificate']}\n"
        plain_ids, plain_start = common.encode_with_boundary(tokenizer, item["prompt"], item["rollout"])
        privileged_ids, privileged_start = common.encode_with_boundary(tokenizer, privileged_prompt, item["rollout"])

        with torch.no_grad():
            plain_teacher = token_logprobs(model, plain_ids.to(device), plain_start)
            privileged_teacher = token_logprobs(model, privileged_ids.to(device), privileged_start)
            span = min(plain_teacher.shape[0], privileged_teacher.shape[0])
            evidence = privileged_teacher[:span] - plain_teacher[:span]
            mask = (evidence > 0).float()
        student = token_logprobs(model, plain_ids.to(device), plain_start)[:span]
        total = total + (mask * student).sum()
        kept += int(mask.sum())
        seen += span
    return grad_of(model, -total / len(batch)), kept / max(seen, 1)


def grad_vcpo(model, tokenizer, batch, device) -> torch.Tensor:
    """VCPO: the proposal-corrected coefficient p_theta * R / D.

    The coefficient is detached, as in the estimator: it weights the score
    function rather than being differentiated through.

    The coefficient is computed in log space and then normalized by the batch
    maximum. ``p_theta = log_p.exp()`` underflows to zero in bf16 for long
    completions, which would zero the whole gradient and make the cosine NaN.
    Cosine is scale-invariant, so normalizing the coefficients preserves the
    direction; the discarded absolute magnitude is exactly what E3 measures.
    """

    log_coeffs = []
    entries = []
    for item in batch:
        ids, start = common.encode_with_boundary(tokenizer, item["prompt"], item["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        # D = n_on_policy * p_theta + q, with q = 1 the normalized certificate
        # proposal. log D = logaddexp(log n + log p_theta, log 1).
        log_D = float(np.logaddexp(np.log(len(batch)) + log_p, 0.0))
        log_c = log_p + (0.0 if item["reward"] > 0 else -np.inf) - log_D
        log_coeffs.append(log_c)
        entries.append((ids, start))

    max_log_c = max(log_coeffs)
    total = 0.0
    for (ids, start), log_c in zip(entries, log_coeffs):
        if log_c == -np.inf:
            continue
        coeff = float(np.exp(log_c - max_log_c))  # in (0, 1]
        total = total + coeff * token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="humaneval", choices=sorted(common.LOADERS))
    parser.add_argument("--n-tasks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="randomly sample n_tasks (robustness check); None = task-id prefix")
    parser.add_argument("--tag", default="")
    parser.add_argument("--spill", action="store_true",
                        help="spill reference gradients to disk (memmap) to bound RAM (for 9B models)")
    args = parser.parse_args()

    common.set_seed(args.seed)
    model, tokenizer = load_model(args.model, train=True)
    model.requires_grad_(True)
    device = next(model.parameters()).device
    batch = build_batch(model, tokenizer, args.n_tasks, device, dataset=args.dataset,
                        shuffle_seed=args.shuffle_seed)

    # The three reference gradients are ~2 bytes/param on CPU (bf16): ~14 GB
    # for a 7B, ~18 GB for a 9B. Holding all three in RAM (~42-54 GB) fits the
    # ~120 GB cgroup cap, so keep them in RAM and skip disk I/O. (A 14B model
    # would be ~84 GB, too close to the cap — use the disk-spill path there.)
    ref_names = ["fq", "sft", "vcpo", "fq_shuffled"]
    refs = {}
    norms = {}
    spill_dir = None
    if args.spill:
        from pathlib import Path
        spill_dir = Path(common.DATA_ROOT) / "tmp_e6"
        spill_dir.mkdir(parents=True, exist_ok=True)
    for ref_name, fn in zip(ref_names, (grad_F_q, grad_sft, grad_vcpo, grad_F_q_shuffled)):
        g = fn(model, tokenizer, batch, device)
        norms[ref_name] = float(g.norm().item())
        if spill_dir is not None:
            path = spill_dir / f"e6_{args.model.split('/')[-1]}_s{args.seed}_{ref_name}.pt"
            torch.save(g, path)  # default zipfile=True: the mmap-compatible format
            refs[ref_name] = torch.load(path, mmap=True)  # stream from disk, not RAM
        else:
            refs[ref_name] = g.cpu()
        del g
        torch.cuda.empty_cache()

    # Does shuffling actually move the reference gradient? The shuffled control
    # is only a control if it points somewhere else. If the two F_q gradients
    # agree, the "correct vs shuffled" comparison has been measuring a quantity
    # against itself, and the EDGE reversal is noise rather than a finding.
    cos_fq_vs_shuffled = cosine(refs["fq"], refs["fq_shuffled"])
    print(f"  meta-check: cos(grad_Fq, grad_Fq_shuffled) = {cos_fq_vs_shuffled:.4f}")

    _ref_key = {"fq": "Fq", "sft": "SFT", "vcpo": "VCPO", "fq_shuffled": "Fq_shuffled"}

    def cos_against_refs(gradient) -> dict:
        row = {}
        for ref_name in ref_names:
            row[f"cos_with_grad_{_ref_key[ref_name]}"] = cosine(gradient, refs[ref_name])
        return row

    results = {}
    edge_mask_rate = None
    for name, fn in (
        ("hsd", grad_hsd),
        ("opid", grad_opid),
        ("edge_opd", grad_edge_opd),
    ):
        try:
            output = fn(model, tokenizer, batch, device)
            gradient, extra = output if isinstance(output, tuple) else (output, None)
            if name == "edge_opd":
                edge_mask_rate = extra
            gradient = gradient.cpu()
            norms[name] = float(gradient.norm().item())
            torch.cuda.empty_cache()
            results[name] = {"implemented": True, **cos_against_refs(gradient)}
            del gradient
        except (RuntimeError, ValueError) as error:
            # Report rather than approximate: a rule that cannot be expressed
            # in this single-turn setting is a scope finding, not a gap to fill.
            results[name] = {"implemented": False, "reason": str(error)}

    # vcpo vs the references (vcpo-vs-vcpo is trivially 1.0). Refs are already
    # in RAM (see above). VCPO is NOT a control (it is a reweighted F_q and
    # unstable across batch size); the real control is the shuffled pairing.
    results["vcpo"] = {
        "cos_with_grad_Fq": cosine(refs["vcpo"], refs["fq"]),
        "cos_with_grad_SFT": cosine(refs["vcpo"], refs["sft"]),
        "cos_with_grad_VCPO": 1.0,
        "cos_with_grad_Fq_shuffled": cosine(refs["vcpo"], refs["fq_shuffled"]),
        "implemented": True,
        "note": "not a control: reweighted F_q, unstable across batch size",
    }
    del refs
    if spill_dir is not None:
        import shutil
        shutil.rmtree(spill_dir, ignore_errors=True)

    meta = RunMeta(
        experiment="e6_alignment",
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        dtype="bfloat16",
        extra={
            "n_tasks": len(batch),
            "n_reward_1": sum(1 for item in batch if item["reward"] == 1.0),
            "n_reward_0": sum(1 for item in batch if item["reward"] == 0.0),
            "edge_opd_mask_keep_rate": edge_mask_rate,
            "scope": "update rules only, not the authors' full systems",
            "control": "shuffled-pairing F_q (deranged prompt-rollout)",
            "cos_fq_vs_fq_shuffled": cos_fq_vs_shuffled,
            "shuffle_seed": args.shuffle_seed,
        },
    ).fill_environment()

    dsuffix = "" if args.dataset == "humaneval" else f"_{args.dataset}"
    path = write_result(f"e6_{args.model.split('/')[-1]}{dsuffix}{args.tag}",
                        meta, {"alignment": results, "grad_l2_norms": norms})

    print(f"model={args.model}  batch={len(batch)} "
          f"(R=1: {meta.extra['n_reward_1']}, R=0: {meta.extra['n_reward_0']})")
    print(f"{'method':<10}{'cos(.,Fq)':>12}{'cos(.,SFT)':>12}{'cos(.,VCPO)':>13}{'cos(.,Fq_shuf)':>15}")
    for name, values in results.items():
        if values.get("implemented"):
            print(f"{name:<10}{values['cos_with_grad_Fq']:>12.4f}"
                  f"{values['cos_with_grad_SFT']:>12.4f}{values['cos_with_grad_VCPO']:>13.4f}"
                  f"{values['cos_with_grad_Fq_shuffled']:>15.4f}")
        else:
            print(f"{name:<10}  not implemented: {values['reason'][:50]}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
