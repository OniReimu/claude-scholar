---
id: REPRO.RANDOM_SEED_DOCUMENTATION
slug: repro-random-seed-documentation
severity: error
locked: false
layer: core
artifacts: [code, text]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

所有实验必须显式设置随机种子（Python: `random.seed()`、`np.random.seed()`、`torch.manual_seed()` + `torch.cuda.manual_seed_all()`），并在论文的 Methods 或 Appendix 中记录种子值。

## Rationale

可复现性是科学研究的基础要求。各大顶会（NeurIPS、ICML、ICLR）的 reproducibility checklist 明确要求记录随机种子。缺少种子设置的实验结果不可复现，缺少种子记录的论文无法通过 reproducibility 审查。

## Check

- **LLM 检查（代码）**: 审查实验代码中是否包含种子设置函数调用（`random.seed()`、`np.random.seed()`、`torch.manual_seed()`、`torch.cuda.manual_seed_all()`，或封装函数如 `set_seed()`）
- **LLM 检查（论文）**: 审查论文 Methods 或 Appendix 中是否提及种子值或可复现性声明

## Examples

### Pass

```python
# utils/seed.py
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

```latex
% 论文中记录种子信息
All experiments use random seed 42. Final results are averaged
over 5 independent runs with seeds \{42, 123, 456, 789, 1024\}.
```

### Fail（缺少种子设置和记录）

```python
# train.py — 无任何种子设置
model = MyModel()
trainer = Trainer(model, train_loader)
trainer.train()
# 违规：未调用任何种子设置函数，实验不可复现
```

```latex
% 论文中未提及种子或可复现性
We train our model for 100 epochs using Adam optimizer.
% 违规：未记录种子值，无法通过 reproducibility checklist
```
