---
id: REPRO.COMPUTE_RESOURCES_DOCUMENTED
slug: repro-compute-resources-documented
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

实验部分必须记录计算资源信息：GPU 型号、总 GPU 小时数、每次运行训练时间、以及环境规格（CUDA 版本、主要库版本）。

## Rationale

计算资源信息帮助读者评估方法的实际可行性，也是多个顶会 reproducibility checklist 的必填项。缺少资源信息使读者无法判断方法是否在其硬件条件下可行，也无法准确估计复现成本。

## Check

- **LLM 检查**: 审查 Experiments 或 Appendix 中是否包含以下关键信息：GPU 型号、训练时间（总 GPU 小时数或单次运行时间）、环境规格（CUDA 版本、框架版本）
- **要点**: 至少包含 GPU 型号和训练时间两项，环境规格强烈建议提供

## Examples

### Pass

```latex
\paragraph{Compute Resources.}
All experiments are conducted on 4$\times$ NVIDIA A100 (80\,GB).
Each training run takes approximately 6 hours, totaling
$\sim$48 GPU hours for all experiments (including ablations).
Our environment uses PyTorch 2.1, CUDA 12.1, and Python 3.10.
```

### Fail（缺少计算资源信息）

```latex
\section{Experiments}
We evaluate our method on three benchmarks.
Table~\ref{tab:main} shows the results.
Our method achieves state-of-the-art performance.
% 违规：整个实验部分和附录均未提及 GPU 型号、
% 训练时间或环境规格等计算资源信息
```
