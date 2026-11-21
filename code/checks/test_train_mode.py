"""Compare sampling in train=True vs train=False mode (gradient_checkpointing suspect)."""
import os
import sys
sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model
import e1_formal as E
from datasets import load_dataset

common.set_seed(0)
rows = list(load_dataset("openai/openai_humaneval", split="test"))[:3]

for train_mode in (False, True):
    model, tok = load_model("Qwen/Qwen3.5-0.8B", train=train_mode)
    dev = next(model.parameters()).device
    npass = 0
    print(f"\n===== train={train_mode} =====")
    for r in rows:
        outs = E.sample_rollouts(model, tok, r["prompt"], dev, n=4)
        ok = [E.run_humaneval_test(r["prompt"], o, r["test"], r["entry_point"]) for o in outs]
        npass += sum(ok)
        print(f"{r['task_id']:16s} pass={sum(ok)}/4  first={outs[0][:80]!r}")
    print(f"train={train_mode}: pass {npass}/{len(rows)*4} = {npass/(len(rows)*4):.0%}")
    del model
    import torch; torch.cuda.empty_cache()
