---
name: policy-rule-creator
description: |
  This skill should be used when the user wants to "add a new rule", "create a policy rule", "add a writing rule", "create a lint rule", "add a new check to the policy engine", or needs to extend the policy engine with new rule cards in the claude-scholar framework.
version: 1.0.0
tags: [Meta, Policy, Rules]
---

# Policy Rule Creator

引导用户在 claude-scholar Policy Engine 中创建新的论文写作规则。自动完成 rule card 创建、Registry 注册、skill 集成标记、lint 配置、验证全流程。

## 前置知识

- 规则规范和完整注册表：`policy/README.md`
- 所有 rule card 位于 `policy/rules/`（单一真相源）
- 字段模板和速查：`references/rule-card-template.md`

## 工作流

### Phase 1: 需求收集

通过提问明确以下信息（一次不超过 3 个问题）：

1. **规则内容**：这条规则要求什么？（用一句祈使句描述）
2. **规则分类**：属于哪个领域？（图表 FIG / 表格 TABLE / LaTeX / 论文结构 PAPER / 实验 EXP / 行文 PROSE / 投稿 SUBMIT / 其他）
3. **严重程度**：必须遵守（error）还是建议遵守（warn）？
4. **适用范围**：所有论文（core）、特定领域（domain）、还是特定会议（venue）？
5. **可否覆盖**：不同会议/领域是否需要不同参数？（决定 locked 和 params）
6. **自动化检查**：能否用正则检查？如果能，给出 pattern 和检查逻辑

如果用户已经提供了足够信息，跳过已回答的问题。

### Phase 2: 生成 Rule Card

1. 读取 `references/rule-card-template.md` 获取模板和字段速查
2. 确定 Rule ID 命名：
   - 检查 `policy/README.md` 的 Rule ID Registry，确认 ID 不重复
   - 遵循 `CATEGORY.RULE_NAME` 命名规范（大写、下划线分隔、点号做命名空间）
3. 确定 slug：从 Rule ID 转换为 kebab-case（如 `FIG.NO_IN_FIGURE_TITLE` → `fig-no-in-figure-title`）
4. 填充所有 frontmatter 字段
5. 撰写四个必填 body section：
   - `## Requirement`：祈使句约束声明
   - `## Rationale`：解释规则存在的原因，帮助 LLM 在边界情况判断
   - `## Check`：具体验证方法
   - `## Examples`：Pass 和 Fail 各至少一个代码块（LaTeX 规则用 `latex`，行文/引文规则可用纯文本）
6. 写入 `policy/rules/<slug>.md`

### Phase 3: 注册到 Registry

在 `policy/README.md` 的 `## Rule ID Registry` 表格中，按 Rule ID 字母序插入一行（同 prefix 的规则应相邻）：

```
| RULE.ID | slug | layer | severity | locked | enforcement |
```

去重约束：
- 插入前先检查 `RULE.ID` 是否已存在于 Registry。
- 若已存在，仅更新该行字段（slug/layer/severity/locked/enforcement），不要追加新行。

推荐查重命令（`rg`）：
```bash
RULE_ID="FIG.NO_IN_FIGURE_TITLE"
rg -n "^\|[[:space:]]*${RULE_ID}[[:space:]]*\|" policy/README.md
```

### Phase 4: 添加 Integration Marker

确定哪些 skill 需要引用此规则。

**必须满足 L2 orphan 约束**：至少在一个 entry skill 中放置 marker。当前 entry skills 为（`policy/validate.sh` Check 9 硬编码）：
- `skills/ml-paper-writing/SKILL.md`
- `skills/paper-self-review/SKILL.md`
- `skills/using-claude-scholar/SKILL.md`

步骤：

1. 根据 `phases` 字段定位相关 skill（至少选一个 entry skill）：
   - `ideation` → `skills/research-ideation/SKILL.md`
   - `writing-*` → `skills/ml-paper-writing/SKILL.md` **(entry)**
   - `self-review` → `skills/paper-self-review/SKILL.md` **(entry)**
   - `revision` → `skills/review-response/SKILL.md`
   - `camera-ready` → `skills/ml-paper-writing/SKILL.md` **(entry)**
   - 如果 phases 不含上述任何映射，在 `skills/using-claude-scholar/SKILL.md` **(entry)** 中添加
2. 在相关 skill 工作流的对应步骤中添加 HTML 注释标记：
   ```
   <!-- policy:{rule_id} -->
   ```
   其中 `{rule_id}` 替换为实际的 Rule ID（如 `FIG.NO_IN_FIGURE_TITLE`）
3. 去重约束：添加前先搜索 `policy:{rule_id}`；若该文件已存在 marker，只补 one-liner 文本，不重复添加 marker
4. 如果不确定放在哪个步骤，询问用户

推荐查重命令（`rg`）：
```bash
RULE_ID="FIG.NO_IN_FIGURE_TITLE"
rg -n "policy:${RULE_ID}" skills/ commands/
```

### Phase 5: 配置 Lint（仅 check_kind: regex）

如果规则可用正则检查：

1. 设置 `check_kind: regex`、`enforcement: lint_script`
2. 填写 `lint_patterns`：
   - `pattern`：YAML 格式的正则（注意双转义 `\\\\`）
   - `mode`：`match`（匹配即违规）/ `count`（超阈值违规）/ `negative`（缺失即违规）
   - `threshold` 和 `threshold_param`（count 模式时）
3. 填写 `lint_targets`：目标文件 glob（如 `**/*.tex`）
4. 如果有可覆盖参数，在 `params` 中声明默认值

如果不可用正则检查：
- 设置 `check_kind` 为 `llm_semantic` / `llm_style` / `manual`
- 设置 `enforcement: doc`
- 不填 `lint_patterns` 和 `lint_targets`

### Phase 6: 验证与测试

运行验证脚本确认无回归：

```bash
bash policy/validate.sh
```

全部 PASS 后，如果规则有 lint_patterns，再跑 lint 测试：

```bash
# 对包含 Pass/Fail 样例的 .tex 文件
bash policy/lint.sh --rule RULE.ID path/to/test/
```

验证预期：
- Pass 示例不触发违规
- Fail 示例触发违规
- 如果有 profile override，加 `--profile` 测试覆盖行为

### Phase 7: Profile 更新（如需要）

如果新规则的 `locked: false` 且不同领域/会议需要不同参数：

1. 读取 `policy/profiles/` 下的现有 profile
2. 在相关 profile 的 `## Includes` 列表中添加规则文件路径：
   ```
   - `policy/rules/<slug>.md`
   ```
3. 在 `## Overrides` 表格中添加参数覆盖行：
   ```
   | RULE.ID | params.key | new_value | 原因 |
   ```

去重约束：
- `## Includes` 中若已存在 `policy/rules/<slug>.md`，不重复添加。
- `## Overrides` 中若已存在同一 `RULE.ID + params.key`，更新值与原因，不追加重复行。

推荐查重命令（`rg`）：
```bash
RULE_ID="FIG.FONT_GE_24PT"
SLUG="fig-font-ge-24pt"
PARAM_KEY="min_font_pt"
rg -n "policy/rules/${SLUG}\\.md" policy/profiles/*.md
rg -n "^\|[[:space:]]*${RULE_ID}[[:space:]]*\|[[:space:]]*params\\.${PARAM_KEY}[[:space:]]*\|" policy/profiles/*.md
```

## 常见误匹配与修正

- `RULE.ID` 含 `.`（如 `FIG.NO_IN_FIGURE_TITLE`）时，正则中的点需写成字面量（`\\.`）或直接用 `rg -F` 做纯文本匹配。
- 表格查重建议加行锚点（`^`）和列分隔符（`\|`），避免把注释/正文里的同名字符串误判为已有表项。
- `params` 查重要带完整键（如 `params.min_font_pt`），不要只搜 `min_font_pt`，避免被其他字段“误命中”。
- marker 查重建议统一模式 `policy:[A-Z][A-Z._0-9]*`，避免遗漏带数字的 Rule ID。
- 对包含反斜杠的模式（LaTeX）优先先在小样本上试跑，再写入 rule card，避免 YAML 与 regex 双重转义错误。

## 最小排错流程（validate 过但 lint 结果异常）

当 `bash policy/validate.sh` 已通过，但 `bash policy/lint.sh` 与预期不一致时，按以下顺序排查：

1. **先锁定规则**：仅跑单条规则，缩小范围。
   ```bash
   bash policy/lint.sh --rule RULE.ID path/to/target
   ```
2. **确认目标文件命中**：检查 `lint_targets` 是否真的覆盖到预期文件。
   ```bash
   rg -n "lint_targets:" policy/rules/<slug>.md
   find path/to/target -type f | rg "\\.tex$|\\.md$|\\.py$"
   ```
3. **确认模式与模式类型**：核对 `pattern` / `mode` / `threshold` / `threshold_param` 是否一致。
   ```bash
   rg -n "lint_patterns:|pattern:|mode:|threshold|threshold_param" policy/rules/<slug>.md
   ```
4. **检查 profile 覆盖是否生效**：特别是 `params.*` 与 `locked` 的组合。
   ```bash
   bash policy/lint.sh --rule RULE.ID --profile policy/profiles/<name>.md path/to/target
   rg -n "^\|[[:space:]]*RULE.ID[[:space:]]*\|" policy/profiles/<name>.md
   ```
5. **隔离最小样本复现**：用单文件最小样本验证 Pass/Fail，确认规则语义再回到真实项目。
   ```bash
   mkdir -p /tmp/policy-lint-mini && cp path/to/example.tex /tmp/policy-lint-mini/
   bash policy/lint.sh --rule RULE.ID /tmp/policy-lint-mini
   ```
6. **CI 统一失败标准**：需要将 warning 也视为失败时，使用 `--strict-warn`。
   ```bash
   bash policy/lint.sh --strict-warn path/to/target
   # 或结合 profile / 单规则
   bash policy/lint.sh --strict-warn --profile policy/profiles/<name>.md --rule RULE.ID path/to/target
   ```

## 输出清单

每次创建完成后，展示总结：

```
📋 新规则创建完成
- Rule ID: CATEGORY.RULE_NAME
- 文件: policy/rules/<slug>.md
- Layer: core/domain/venue | Severity: error/warn | Locked: true/false
- Lint: ✅ 自动检查 / ❌ 仅文档约束

📊 变更文件
1. policy/rules/<slug>.md（新建）
2. policy/README.md（Registry 新增一行）
3. skills/xxx/SKILL.md（Integration marker）

✅ 验证结果
- validate.sh: X/12 PASS
- lint.sh: [测试结果]
```
