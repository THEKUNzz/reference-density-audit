# results/ — 实验结果

各实验的结果 JSON，按服务器分目录。

| 目录 | 内容 |
|---|---|
| `results/`（顶层） | S1（Qwen3 / Qwen3.5）的 E1/E2c/E3/E6 结果 |
| `results/server2/` | S2（OLMo / GLM / Qwen3.5 补 seed）的 E1/E3/E6 结果 |
| `results/verified/` | 论文引用的权威结果，`RESULTS_AUDIT.json` 的 manifest 按 sha256 校验这批文件 |
| `results/e7/` | E7 k 子集聚合对照，18 格（6 模型 × 3 数据集） |

## 文件命名

- `e1formal_{model}_seed{0,1,2}.json` — E1 正式结果（按 seed）
- `e3_{model}_{dataset}.json` — E3 衰减（humaneval/mbpp/quixbugs）
- `e6_{model}.json` — E6 梯度对齐
- `e6_length_strata.json` — E6 长度分层分析
- `e7/e7_ksubset_{model}_{dataset}_seed0.json` — E7：完整 Gram、逐证书精确标量、
  k 子集曲线、以及全部验收门的数值

## E7 读法

每个文件的 `gates` 字段记录批次一致性（ESS / N 与档案比对）、G1（逐证书余弦，
仅 HumanEval 的 0.8B/2B 有档案可比）、G3（k=N 端点复现 `RESULTS_AUDIT.json`）。
`gram_method` 区分精确 float64 Gram（5 格）与 count-sketch（13 格），后者的误差
由同一次运行的 `sketch_calibration` 对照精确解标定，不是估计值。

## 汇总文档

- `docs/SUMMARY_SERVER2.md` — E3 Table A/B 斜率汇总
- `docs/SUMMARY_SERVER2_E1.md` — E1 跨家族 all-fail 谱系（6 模型 × 3 seeds + Wilson CI）
