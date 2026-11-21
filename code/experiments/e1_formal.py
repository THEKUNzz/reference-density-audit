#!/usr/bin/env python3
"""E1 (FORMAL): does correction remove structural silence, measured empirically.

For each task we sample a GROUP of n on-policy rollouts and RUN the HumanEval
test on each to assign R in {0,1}. A group is "all-fail" if every rollout fails.
We report:
  - the all-fail fraction (where GRPO is structurally silent)
  - that GRPO's gradient is exactly 0 on an all-fail group
  - that VCPO's gradient is non-zero there (the certificate connection)

This reproduces the paper's "99.9% -> 0.0%" claim with real test execution,
not the assumed-all-fail smoke version.
"""
import argparse
import numpy as np
import torch

import vcpo_common as common
from vcpo_common import load_model, encode_with_boundary, write_result, RunMeta
from e6_alignment import token_logprobs, grad_of
from transformers import StoppingCriteria, StoppingCriteriaList


def run_humaneval_test_status(prompt, completion, test, entry_point, timeout=30):
    """Run the HumanEval test; return 'pass' | 'fail' | 'timeout' | 'harness_error'.

    Runs in a SUBPROCESS with a timeout. The completion is model-generated
    untrusted code: in-process ``exec`` lets a stray ``exit()``/``sys.exit()``
    (SystemExit is not caught by ``except Exception``) kill the whole E1 run,
    and an infinite loop would hang it. A subprocess isolates all of that.

    The four statuses keep "the harness broke" separable from "the model got it
    wrong": a timeout means the model wrote a non-terminating program, not that
    it failed the asserts; a harness_error means the E1 number itself is suspect.
    """
    import subprocess
    import sys

    program = prompt + completion + "\n" + test + "\n" + f"check({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=timeout,
        )
        return "pass" if result.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "harness_error"


def run_humaneval_test(prompt, completion, test, entry_point, timeout=30):
    """Bool wrapper (backward compat for e2b). True iff the test passes."""
    return run_humaneval_test_status(prompt, completion, test, entry_point, timeout) == "pass"


# Standard HumanEval stop sequences. A prompt ends mid-function, so the answer
# is the function body alone; models routinely continue into a __main__ block or
# a second definition. That trailing text is not part of the answer and its
# syntax errors or missing imports fail otherwise-correct solutions. Truncating
# here is the evaluation protocol of the original benchmark, not added leniency.
STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__",
                  "\nprint(", "\n#", "\n@", "\nassert ", "\nimport ", "\nfrom "]

# MBPP: the prompt is a bare docstring and the model writes the whole function
# from scratch, so a leading ``import``/``from`` is a legitimate start, NOT a
# signal to stop. The answer usually ends at the next top-level block (a second
# ``if __name__``, a print/assert of its own tests, or a closing docstring).
MBPP_STOP = ["\nif __name__", "\nprint(", "\nassert ", '\n"""', "\n'''"]


def truncate_at_stop(text: str, stops=STOP_SEQUENCES) -> str:
    """Cut a completion where it leaves the function body."""
    cut = len(text)
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


class StopOnSequences(StoppingCriteria):
    """Halt generation once every sequence has left the function body.

    Without this the model writes to max_new_tokens and the stop sequences are
    applied afterwards, so four fifths of the generated tokens are decoded and
    thrown away. Checking during generation is what makes a 512-token ceiling
    affordable: a well-formed answer stops around 80.
    """

    def __init__(self, tokenizer, prompt_len, stops):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.stops = stops

    def __call__(self, input_ids, scores, **kwargs):
        for row in input_ids:
            text = self.tokenizer.decode(row[self.prompt_len:], skip_special_tokens=True)
            if not any(s in text for s in self.stops):
                return False        # at least one sequence is still inside the body
        return True


@torch.no_grad()
def sample_rollouts(model, tokenizer, prompt, device, n=4, max_new=512, stop_sequences=STOP_SEQUENCES):
    """Sample n completions in one batched call, truncated at the body's end.

    The n rollouts share a prompt and are independent, so they go through the
    generator together: batch=1 leaves the GPU badly underused, and this is the
    inner loop of both E1 and E2(b).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    criteria = StoppingCriteriaList([StopOnSequences(tokenizer, prompt_len, stop_sequences)])

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        num_return_sequences=n,          # all n rollouts in one forward pass
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=criteria,
    )
    return [truncate_at_stop(tokenizer.decode(row[prompt_len:], skip_special_tokens=True), stop_sequences)
            for row in gen]


def grad_grpo(model, tokenizer, group, device):
    """GRPO gradient on a group of rollouts (all R=0 -> advantage 0 -> grad 0)."""
    rewards = [g["reward"] for g in group]
    mean_r = sum(rewards) / len(rewards)
    var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    total = 0.0
    for g in group:
        adv = (g["reward"] - mean_r) / std if std > 0 else 0.0
        ids, start = encode_with_boundary(tokenizer, g["prompt"], g["rollout"])
        total = total + adv * token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -total)


def grad_vcpo_conn(model, tokenizer, group, device):
    """VCPO gradient from the certificate connection (R=1), detached coefficient."""
    log_cs = []
    entries = []
    for g in group:
        ids, start = encode_with_boundary(tokenizer, g["prompt"], g["certificate"])
        with torch.no_grad():
            log_p = float(token_logprobs(model, ids.to(device), start).sum())
        log_D = float(np.logaddexp(np.log(len(group)) + log_p, 0.0))
        log_cs.append(log_p - log_D)
        entries.append((ids, start))
    max_log_c = max(log_cs)
    total = 0.0
    for (ids, start), log_c in zip(entries, log_cs):
        coeff = float(np.exp(log_c - max_log_c))
        total = total + coeff * token_logprobs(model, ids.to(device), start).sum()
    return grad_of(model, -total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="humaneval", choices=sorted(common.LOADERS))
    parser.add_argument("--n-tasks", type=int, default=164)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="cap tasks, for smoke runs")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    common.set_seed(args.seed)
    # Sample in EVAL mode: load_model(train=True) enables gradient checkpointing,
    # which corrupts generate() (KV cache / attention) and turns the sampled
    # rollouts into garbage -> a fake 100% all-fail rate. Gradients are computed
    # separately below by switching to train mode only for grad_grpo/grad_vcpo.
    model, tokenizer = load_model(args.model, train=False)
    device = next(model.parameters()).device

    items = common.load_dataset_items(args.dataset, args.limit)[: args.n_tasks]

    def run_test_status(item, completion):
        if args.dataset == "mbpp":
            return common.run_mbpp_test_status(completion, item["test_list"], item["test_imports"])
        return run_humaneval_test_status(item["prompt"], completion, item["test"], item["entry_point"])

    n_all_fail = 0
    n_groups = 0
    n_test_failed = 0
    n_timeout = 0
    n_harness_error = 0
    all_fail_groups = []
    smoke = args.limit is not None
    stops = MBPP_STOP if args.dataset == "mbpp" else STOP_SEQUENCES
    for item in items:
        rollouts = sample_rollouts(model, tokenizer, item["prompt"], device,
                                   n=args.group_size, stop_sequences=stops)
        statuses = [run_test_status(item, r) for r in rollouts]
        rewards = [1.0 if s == "pass" else 0.0 for s in statuses]
        n_test_failed += sum(1 for s in statuses if s == "fail")
        n_timeout += sum(1 for s in statuses if s == "timeout")
        n_harness_error += sum(1 for s in statuses if s == "harness_error")
        if smoke:
            print(f"\n===== task {item['task_id']} =====", flush=True)
            print(f"[prompt]\n{item['prompt']}", flush=True)
            for i, (r, s) in enumerate(zip(rollouts, statuses)):
                print(f"[rollout {i} status={s}]\n{r}", flush=True)
        n_groups += 1
        if all(r == 0.0 for r in rewards):
            n_all_fail += 1
            all_fail_groups.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "rollouts": rollouts,
                "reward": 0.0,  # all-fail
                "certificate": item["certificate"],
            })
        if n_groups % 20 == 0:
            print(f"  {n_groups} groups done, {n_all_fail} all-fail", flush=True)

    all_fail_frac = n_all_fail / n_groups if n_groups else 0.0

    # On a representative all-fail group, verify GRPO=0 and VCPO>0.
    grpo_norm = vcpo_norm = None
    if all_fail_groups:
        # Switch to train mode only for the gradient computation (small batch,
        # so no gradient checkpointing is needed).
        model.train()
        model.requires_grad_(True)
        rep = all_fail_groups[0]
        # make a "group" of the 4 failed rollouts for GRPO
        grpo_group = [{"prompt": rep["prompt"], "rollout": r, "reward": 0.0}
                      for r in rep["rollouts"]]
        g_grpo = grad_grpo(model, tokenizer, grpo_group, device).cpu()
        g_vcpo = grad_vcpo_conn(model, tokenizer, [rep], device).cpu()
        grpo_norm = float(g_grpo.norm())
        vcpo_norm = float(g_vcpo.norm())
        torch.cuda.empty_cache()

    if n_harness_error > 0:
        print(f"  ⚠ WARNING: {n_harness_error} harness errors — all-fail rate is NOT trustworthy", flush=True)

    meta = RunMeta(
        experiment="e1_zero_gradient_formal",
        model=args.model, dataset=args.dataset, seed=args.seed, dtype="bfloat16",
        extra={"n_tasks": n_groups, "group_size": args.group_size,
               "n_all_fail": n_all_fail, "all_fail_fraction": all_fail_frac,
               "n_test_failed": n_test_failed, "n_timeout": n_timeout,
               "n_harness_error": n_harness_error,
               "note": f"real sampling + real {args.dataset} test execution"},
    ).fill_environment()
    payload = {
        "all_fail_fraction": all_fail_frac,
        "n_all_fail": n_all_fail, "n_groups": n_groups,
        "n_test_failed": n_test_failed, "n_timeout": n_timeout,
        "n_harness_error": n_harness_error,
        "grad_grpo_norm": grpo_norm, "grad_vcpo_norm": vcpo_norm,
    }
    dsuffix = "" if args.dataset == "humaneval" else f"_{args.dataset}"
    path = write_result(f"e1formal_{args.model.split('/')[-1]}_seed{args.seed}{dsuffix}{args.tag}", meta, payload)

    print(f"model={args.model}")
    print(f"  all-fail fraction = {all_fail_frac:.4f} ({n_all_fail}/{n_groups} groups)")
    print(f"  |grad GRPO| on all-fail = {grpo_norm:.3e}  (want ~0)")
    print(f"  |grad VCPO| on all-fail = {vcpo_norm:.3e}  (want >0)")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
