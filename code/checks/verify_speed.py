"""Verify the batched + StoppingCriteria sampling is fast and still passes."""
import sys, time
sys.path.insert(0, ".")
import vcpo_common as common
from vcpo_common import load_model
import e1_formal as E
from datasets import load_dataset

model, tok = load_model("Qwen/Qwen3.5-0.8B")
dev = next(model.parameters()).device
rows = list(load_dataset("openai/openai_humaneval", split="test"))[:5]
t = time.time()
npass = 0
for r in rows:
    outs = E.sample_rollouts(model, tok, r["prompt"], dev, n=4)
    npass += sum(E.run_humaneval_test(r["prompt"], o, r["test"], r["entry_point"]) for o in outs)
dt = time.time() - t
print(f"5 tasks x 4 rollouts: {dt:.0f}s  ({dt/5:.1f}s per task)")
print(f"pass {npass}/20 = {npass/20:.0%}   (before: ~120s per task)")
