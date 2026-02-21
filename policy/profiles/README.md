# Policy Profiles 规范

Profile = 领域 + 会议组合，定义加载哪些规则以及参数覆盖。

> v1 限制：`policy/lint.sh --profile` 当前只支持加载单个 profile 文件，不支持 profile inheritance/composition。
> 因此如 `security-sok-sp` 这类 SoK profile 需显式复制 base includes，再追加 `SOK.*` 规则。

---

## Profile Frontmatter

```yaml
---
name: security-neurips
domain: security
venue: neurips
---
```

SoK profile 示例：

```yaml
---
name: security-sok-sp
domain: security
venue: sp
---
```

---

## Body Sections

1. `## Includes` — 显式逐文件列出加载的规则（不用 glob）
2. `## Overrides` — 表格：Rule ID + 字段 + 新值 + 原因
3. `## Domain-Specific Rules` — 领域专属规则（M2 补充完整卡片）
4. `## Venue Quick Facts` — 页数、格式、截止日期
5. `## Cross-References` — 指向 `rules/` 目录中的关联文件

---

## Override 规范

### 允许修改的字段

- `locked: false` 的规则：可覆盖 `severity` 和 `params` 中声明的任意 key
- `locked: true` 的规则：**severity 和 params 均不可覆盖**

### locked 语义

当规则卡片声明 `locked: true` 时：
- 该规则的 `severity` 在所有 profile 中固定不变
- 该规则的 `params`（如有）在所有 profile 中固定不变
- 如 profile 尝试覆盖 locked 规则，该 override 行视为**无效**（LLM 应忽略并提示 warning）

### Override 表格格式

```markdown
| Rule ID | 字段 | 新值 | 原因 |
|---------|------|------|------|
| RULE.ID | severity | error | 理由说明 |
| RULE.ID | params.key_name | 28 | 理由说明 |
```

### 约束

- Override 引用的 `params.*` key 必须在规则卡片的 `params` 字段中已声明
- Override 引用的 Rule ID 必须在 `## Includes` 列表中
- 不可覆盖 `locked: true` 规则的任何字段
