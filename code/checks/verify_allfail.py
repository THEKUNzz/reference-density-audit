"""Is the 100% all-fail rate real, or another harness bug?

Two possibilities look identical in the aggregate number: the model genuinely
cannot solve these tasks (which is exactly the sparse-reward regime VCPO
targets), or the judging pipeline is still broken (as the 128-token truncation
was). They are told apart by looking at what the model actually writes.
"""
import os
import sys
sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model
import e1_formal as E
from datasets import load_dataset
import random

model, tok = load_model("Qwen/Qwen3.5-0.8B")
dev = next(model.parameters()).device
rows = list(load_dataset("openai/openai_humaneval", split="test"))
common.set_seed(0)

# Sample across the whole benchmark, not just the easy opening tasks.
picked = random.Random(0).sample(rows, 12)
npass = 0
for r in picked:
    outs = E.sample_rollouts(model, tok, r["prompt"], dev, n=4)
    ok = [E.run_humaneval_test(r["prompt"], o, r["test"], r["entry_point"]) for o in outs]
    npass += sum(ok)
    print(f"{r['task_id']:16s} pass={sum(ok)}/4")
    if not any(ok):
        print(f"  sample output: {outs[0][:150]!r}")
print(f"\nper-rollout pass rate: {npass}/{len(picked)*4} = {npass/(len(picked)*4):.1%}")
