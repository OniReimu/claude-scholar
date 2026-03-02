# Workflow Orchestrator

> 单模式、可恢复的研究 run 协调层

## 概述

Workflow Orchestrator 是 Claude Scholar 的薄运行时层，为研究流程提供：

1. **Run State** — 持久化记录当前研究 run 的阶段、产物、决策
2. **Stage Gates** — 在关键节点要求用户审批后才继续
3. **Resume + Invalidation** — session start 时比对产物指纹，变更时自动标记阶段为 stale，仅重跑必要检查
4. **Status Surfacing** — session 启动时自动展示当前进度并检测 stale 阶段

Orchestrator 不引入新命令。用户继续使用自然语言提示和现有命令（`/research-init`、`/analyze-results` 等），orchestrator 在后台透明运行。

## 存储布局

```
.claude/orchestrator/
├── active-run.json          # 指向当前活跃 run
└── runs/
    └── <run_id>/
        ├── run.json          # run 状态（阶段、产物、输入）
        └── events.ndjson     # 事件日志（NDJSON 格式）
```

产物文件保持在项目原有位置（项目根目录、`analysis-output/` 等），run state 只存储相对路径和 hash 指纹。

## Run ID 生成

格式：`YYYYMMDD-HHmmss-<4位随机hex>`

示例：`20260302-143052-a7f3`

## 阶段模型 (v1)

| 序号 | Stage ID | 说明 | 可选 |
|------|----------|------|------|
| 1 | `intake` | 范围、约束、目标 venue/profile | 否 |
| 2 | `literature` | 文献检索 + 差距分析 | 否 |
| 3 | `proposal` | 研究问题 + 方法计划 | 否 |
| 4 | `development` | 代码架构 + 基础设施 | 否 |
| 5 | `experiments` | 实验计划 + 人工执行 | 否 |
| 6 | `analysis` | 结果分析 + 可视化 | 否 |
| 7 | `writeup` | 论文撰写 | 否 |
| 8 | `self_review` | 自审 + policy lint | 否 |
| 9 | `rebuttal` | 审稿回复 | 是 |
| 10 | `post_acceptance` | 录用后处理 | 是 |

## 阶段状态枚举

| Status | 含义 |
|--------|------|
| `pending` | 未开始 |
| `in_progress` | 进行中 |
| `blocked` | 等待用户输入/操作（常见于 `experiments`） |
| `done` | 已完成，产物已指纹化 |
| `stale` | 产物/输入变更导致失效，需重新检查 |

## 完成判定

一个阶段被标记为 `done` 需要满足：

1. `expected_artifacts` 中所有产物存在且已指纹化
2. 所有 `gates` 通过（人工审批 + policy 检查）
3. 状态写入 `run.json` 并追加事件到 `events.ndjson`

## 失效 (Invalidation) 机制

当已完成阶段的产物文件 hash 与记录的指纹不一致时：

1. Session start hook 自动比对 `done` 阶段的 `artifacts.*.fingerprints` 与当前文件 hash
2. 检测到 mismatch 时调用 `setStageStatus(stale)`，该阶段标记为 `stale`
3. `setStageStatus` 同时清除下游 `done` 阶段为 `pending`
4. 仅需重跑受影响阶段的 gates 检查，无需重做全部工作
5. Hook 在输出中提示用户哪些阶段变为 stale 及本次新检测的数量

## 实验阶段的特殊处理

`experiments` 阶段由人类执行：

1. 阶段进入 `blocked` 状态，附带明确指示："需要运行什么 + 输出写入何处"
2. 用户完成实验后提供 `data_path`
3. 系统验证 `data_path` 存在，对结果文件生成指纹
4. 用户确认结果为 `ACTUAL_RUN`（非 `FABRICATED_PLACEHOLDER`）
5. 阶段标记为 `done`

## 回滚 (Rollback)

- 任何阶段可通过用户指令回滚到 `pending`（"roll back to stage X"）
- Orchestrator 记录回滚事件
- 该阶段下游的所有 `done` 状态被清除（标记为 `pending`）

## 并发预期

v1 推荐单活跃 session。library 提供 `withRunLock()` 尽力型 advisory locking（lockfile `wx` 模式），但 **调用方需主动使用**——`updateRun`/`markStage` 本身不自动加锁。多 session 并发写入可能产生竞态条件。

## Hooks 集成

### Session Start (`hooks/session-start.js`)

展示：
- 活跃 run ID + 标题
- 当前阶段 + 状态
- 建议的下一步操作

### Skill Forced Eval (`hooks/skill-forced-eval.js`)

在 skill 列表前注入：
- 活跃 run ID + 当前阶段
- 一行指令："如果请求与活跃 run 相关，从当前阶段继续并更新 run 状态"

## run.json Schema

```json
{
  "id": "20260302-143052-a7f3",
  "title": "RLHF Robustness Study",
  "profile": "security-sok-sp",
  "created_at": "2026-03-02T14:30:52Z",
  "updated_at": "2026-03-02T15:10:00Z",
  "current_stage": "literature",
  "stages": {
    "intake": { "status": "done", "started_at": "...", "completed_at": "..." },
    "literature": { "status": "in_progress", "started_at": "..." }
  },
  "inputs": {
    "venue": "NeurIPS",
    "profile": "security-sok-sp"
  },
  "artifacts": {
    "literature": {
      "fingerprints": {
        "literature-review.md": "sha256:abc123...",
        "references.bib": "sha256:def456..."
      }
    }
  },
  "gate_results": {}
}
```

## 文件引用

- 阶段注册表：`orchestrator/stages.json`
- Run Card 合约：`orchestrator/run-card.md`
- 运行时库：`scripts/lib/orchestrator.js`
