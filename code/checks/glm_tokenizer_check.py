#!/usr/bin/env python3
"""GLM tokenizer 边界核对（只 decode，不跑梯度）。

对每个 model × {clean, mismatched} × 64 行，检查 encode_with_boundary 的 start
是否正好把 prompt 和 completion 切开。判据：exact_match 与 drift_chars。

错配逐字照抄 grad_F_q_shuffled 的 seed=0 deranged 配对。completion 用 rollout
（R=1 = 证书，R=0 = 贪心生成），与 grad_F_q / grad_F_q_shuffled 完全同源。

用法：python glm_tokenizer_check.py <model_id> [<model_id> ...]
结果 append 到 data/results/glm_tokenizer_check.json（12 条）。
"""
import os
import sys
import random
import json
import torch

sys.path.insert(0, os.environ.get("EXPERIMENT_CODE", "."))
import vcpo_common as common
from vcpo_common import load_model, encode_with_boundary
from e6_alignment import build_batch

OUT = "data/results/glm_tokenizer_check.json"
N_TASKS = 64


def deranged_pairing(batch, seed=0):
    """与 grad_F_q_shuffled 完全相同的错配：seed=0 deranged 配对，作用在 mixed rollout 上。"""
    rng = random.Random(seed)
    rollouts = [item["rollout"] for item in batch]
    for _ in range(100):
        rng.shuffle(rollouts)
        if all(r != item["rollout"] for r, item in zip(rollouts, batch)):
            break
    return [dict(item, rollout=r) for item, r in zip(batch, rollouts)]


def check_row(tokenizer, prompt, completion, row_idx):
    ids, start = encode_with_boundary(tokenizer, prompt, completion)
    ids_list = ids[0].tolist()
    decoded_prefix = tokenizer.decode(ids_list[:start], skip_special_tokens=True)
    decoded_suffix = tokenizer.decode(ids_list[start:], skip_special_tokens=True)
    exact = (decoded_prefix == prompt) and (decoded_suffix == completion)
    drift = len(decoded_prefix) - len(prompt)
    return {
        "row": row_idx,
        "start": int(start),
        "prompt": prompt,
        "decoded_prefix": decoded_prefix,
        "decoded_suffix": decoded_suffix,
        "exact_match": bool(exact),
        "drift_chars": int(drift),
    }


def summarize(rows):
    n = len(rows)
    exact = sum(1 for r in rows if r["exact_match"])
    drift = [r["drift_chars"] for r in rows]
    nonzero = [d for d in drift if d != 0]
    mism = [r for r in rows if not r["exact_match"]]
    mism.sort(key=lambda r: -abs(r["drift_chars"]))
    examples = []
    for r in mism[:3]:
        examples.append({
            "row": r["row"],
            "drift_chars": r["drift_chars"],
            "prompt_tail_50": r["prompt"][-50:],
            "decoded_prefix_tail_50": r["decoded_prefix"][-50:],
            "decoded_suffix_head_50": r["decoded_suffix"][:50],
        })
    return {
        "n_rows": n,
        "n_exact_match": exact,
        "n_mismatch": n - exact,
        "drift_chars_distribution": {
            "min": min(drift) if drift else 0,
            "max": max(drift) if drift else 0,
            "nonzero_count": len(nonzero),
        },
        "example_mismatches": examples,
    }


def probe_one(model_id):
    common.set_seed(0)
    model, tokenizer = load_model(model_id, train=False)
    device = next(model.parameters()).device
    batch = build_batch(model, tokenizer, N_TASKS, device, dataset="humaneval", shuffle_seed=0)
    del model
    torch.cuda.empty_cache()
    shuffled = deranged_pairing(batch, seed=0)

    clean_rows = [check_row(tokenizer, it["prompt"], it["rollout"], i) for i, it in enumerate(batch)]
    mism_rows = [check_row(tokenizer, it["prompt"], shuffled[i]["rollout"], i) for i, it in enumerate(batch)]

    clean_rec = {"model": model_id.split("/")[-1], "text_condition": "clean"}
    clean_rec.update(summarize(clean_rows))
    mism_rec = {"model": model_id.split("/")[-1], "text_condition": "mismatched"}
    mism_rec.update(summarize(mism_rows))

    print(json.dumps(clean_rec, ensure_ascii=False), flush=True)
    print(json.dumps(mism_rec, ensure_ascii=False), flush=True)
    return clean_rec, mism_rec


def main():
    model_ids = sys.argv[1:]
    try:
        with open(OUT) as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        records = []
    for mid in model_ids:
        try:
            c, m = probe_one(mid)
            records.append(c)
            records.append(m)
        except Exception as e:
            print(json.dumps({"model": mid.split("/")[-1], "error": str(e)[:200]}), flush=True)
            records.append({"model": mid.split("/")[-1], "error": str(e)[:200]})
        with open(OUT, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"WROTE {OUT} ({len(records)} records)", flush=True)


if __name__ == "__main__":
    main()
