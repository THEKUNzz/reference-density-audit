# data/ — 本地数据集

| 文件 | 内容 | 来源 |
|---|---|---|
| `quixbugs.jsonl` | QuixBugs 数据集 | 由 `experiments/llm/prepare_quixbugs.py` 生成 |

HumanEval / MBPP 数据集通过 HuggingFace 缓存加载（不落本地文件），见
`experiments/llm/vcpo_common.py` 的 `load_model` 与各实验脚本的 `load_dataset`。
