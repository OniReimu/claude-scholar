# Orchestrator Run Card

> Skills 和 Agents 与 Orchestrator 交互的规范合约

## 状态存储

```
.claude/orchestrator/
├── active-run.json                 # { "run_id": "...", "activated_at": "..." }
└── runs/<run_id>/
    ├── run.json                    # 完整 run 状态
    └── events.ndjson               # 事件日志
```

## run.json Schema

```json
{
  "id": "20260302-143052-a7f3",
  "title": "Research Title",
  "profile": "security-sok-sp",
  "venue": "NeurIPS",
  "created_at": "2026-03-02T14:30:52Z",
  "updated_at": "2026-03-02T15:10:00Z",
  "current_stage": "literature",
  "stages": {
    "<stage_id>": {
      "status": "pending|in_progress|blocked|done|stale",
      "started_at": "...",
      "completed_at": "...",
      "note": "optional note"
    }
  },
  "inputs": {},
  "artifacts": {
    "<stage_id>": {
      "tracked_files": ["<path-a>", "<path-b>"],
      "fingerprints": { "<path>": "sha256:..." }
    }
  },
  "gate_results": {
    "<stage_id>": {
      "last_run": "...",
      "passed": true,
      "summary": "..."
    }
  }
}
```

## 阶段生命周期操作

### 阶段开始

当 skill/agent 开始处理其所属阶段时：

1. **加载活跃 run**：调用 `loadActiveRun({ cwd })` 检查是否存在活跃 run
2. **如果无活跃 run**：调用 `initRun({ cwd, title, venue, profile })` 初始化
3. **标记 `in_progress`**：调用 `markStage({ cwd, stageId, status: 'in_progress' })`
4. **记录输入/假设**：调用 `updateRun({ cwd, patch: { inputs: { ... } } })`

### 阶段结束

当 skill/agent 完成其所属阶段时：

1. **写入产物**：将文件输出到预期路径
2. **收集 + 指纹化产物**：优先调用 `fingerprintStageArtifacts({ cwd, run, stageId, extraPaths })`
   - 该 helper 会自动纳入 stage contract 中声明的 `kind:file` 产物
   - `writeup` 阶段会从 `artifacts.writeup.main_tex` 自动展开本地 `\input` / `\include` / bibliography / `\includegraphics` 依赖
   - `extraPaths` 只用于补充，不能替代 contract baseline
   - **不要指纹化大体积原始数据**（实验结果 CSV、模型 checkpoint、图片原始文件等），否则 SessionStart 指纹比对会拖慢启动
   - 经验法则：单个被指纹化文件不应超过 1 MB
3. **保存指纹**：调用 `updateRun({ cwd, patch: { artifacts: { [stageId]: { tracked_files, fingerprints } } } })`
4. **执行 gates**（如有）：运行 policy lint 或请求人工审批
5. **保存 gate 结果**：调用 `updateRun({ cwd, patch: { gate_results: { [stageId]: { ... } } } })`
6. **请求审批**：向用户确认产物质量
7. **标记 `done`**：调用 `markStage({ cwd, stageId, status: 'done' })`

## Polish 模式（改稿合约）

### 初始化

```javascript
const run = orch.initPolishRun({
  cwd,
  title: 'Polish: Paper Title',
  mainTexPath: 'paper/main.tex',
  profile: 'security-neurips',  // optional
  venue: 'NeurIPS',             // optional
});
```

### 阶段流转

1. `intake` → 自动 `done`（polish mode intake）
2. `literature` ~ `analysis` → 自动 `skipped`
3. `writeup` → 自动 `done`（已有 draft）
4. `self_review` → `in_progress`（开始检查）
5. 有违规 → `rewrite`（逐 section 改写）→ 回到 `self_review`
6. 全部 pass → `rebuttal` 或结束

### Rewrite 阶段的合约

Rewrite 阶段必须：
1. 读取 `artifacts.self_review.violation_report`
2. 加载 `policy/style-guide.md`（⚠️ MANDATORY）
3. 对每个违规 section 进行改写
4. 记录 `artifacts.rewrite.sections_rewritten` 和 `artifacts.rewrite.iteration`
5. 请求人工审批后才能回到 `self_review`

## 失效规则 (Invalidation)

Session start hook 自动比对 `done` 阶段的 `artifacts.*.fingerprints` 与当前文件 hash。当 mismatch 被检测到时：

1. Hook 调用 `setStageStatus(stale)` 标记该阶段
2. `setStageStatus` 同时清除所有下游 `done` 阶段为 `pending`，并同步 `current_stage`
3. 仅需重跑受影响阶段的 gates 检查
4. Hook 输出中提示用户 stale 阶段及本次新检测的数量

## 人工实验边界 (experiments stage)

`experiments` 阶段的特殊处理流程：

1. **进入 `blocked`**：附带明确说明"需要运行什么 + 输出写入何处"
2. **用户完成实验**：提供 `data_path`（结果文件/目录路径）
3. **验证 `data_path` 存在**：检查文件/目录是否存在
4. **指纹化结果**：对选定的结果文件生成 hash
5. **用户确认数据来源**：
   - `ACTUAL_RUN` — 真实实验结果
   - `FABRICATED_PLACEHOLDER` — 占位/模拟数据（需明确标注）
6. **标记 `done`**

## 回滚规则 (Rollback)

- 用户可通过指令 "roll back to stage X" 回滚任意阶段
- 调用 `setStageStatus({ cwd, stageId, status: 'pending', reason })` 执行回滚
- 该阶段下游的所有 `done` 状态被清除为 `pending`
- 回滚事件记录到 `events.ndjson`
- 回滚不删除已有产物文件，仅重置状态

## Library API

运行时库路径：`scripts/lib/orchestrator.js`

```javascript
const orch = require('./scripts/lib/orchestrator');

// 核心操作
orch.loadActiveRun({ cwd })           // 加载活跃 run
orch.initRun({ cwd, title, ... })     // 初始化新 run
orch.markStage({ cwd, stageId, status, note })  // 标记阶段状态
orch.updateRun({ cwd, patch })        // 更新 run 字段
orch.fingerprintFiles({ cwd, paths }) // 文件指纹化
orch.collectTrackedFiles({ cwd, run, stageId, extraPaths }) // 收集应追踪文件
orch.fingerprintStageArtifacts({ cwd, run, stageId, extraPaths }) // 生成 tracked_files + fingerprints
orch.appendEvent({ cwd, runId, type, payload })  // 追加事件
orch.setStageStatus({ cwd, stageId, status, reason })  // 设置状态（支持回滚）
orch.withRunLock({ cwd, runId }, fn)  // Advisory locking
```
