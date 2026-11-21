"""Export the audited alignment pairs used by the Figure 2 PGFPlots source."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = json.loads((ROOT / "vcpo-paper" / "RESULTS_AUDIT.json").read_text())
OUT = ROOT / "vcpo-paper" / "figures"

MODELS = [
    "Qwen3.5-0.8B",
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "OLMo-2-1124-7B",
    "glm-4-9b",
]
DATASETS = ["humaneval", "mbpp", "quixbugs"]
COMPONENTS = [("hsd", "ce"), ("opid", "shift"), ("edge_opd", "mask")]


def main():
    lookup = {(row["dataset"], row["model"]): row for row in AUDIT["alignment"]}
    rows = {name: [] for _, name in COMPONENTS}
    x = 0
    for dataset in DATASETS:
        for model in MODELS[:-1]:
            for key, name in COMPONENTS:
                x += 1
                values = lookup[(dataset, model)]["alignment"][key]
                gc = values["cos_with_grad_VCPO"]
                fq = values["cos_with_grad_Fq"]
                rows[name].append((x, gc, fq, fq - gc))

    OUT.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        path = OUT / f"alignment_pairs_{name}.dat"
        text = "x gc fq delta\n" + "\n".join(
            f"{x} {gc:.9f} {fq:.9f} {delta:.9f}" for x, gc, fq, delta in values
        )
        path.write_text(text + "\n")
    print("Wrote 45 plotted pairs; all 9 GLM values remain in the appendix table")


if __name__ == "__main__":
    main()
