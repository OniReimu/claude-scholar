# Provider 配置指南

图像生成 API 的配置方法。至少需要配置其中一个 provider。

---

## 环境变量配置

在项目根目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# 至少配置一个 API key
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key

# 可选：指定默认 provider（默认自动检测，Google 优先）
DEFAULT_PROVIDER=google
```

优先级：`process.env` > 项目根目录 `.env`

---

## Google Gemini（推荐）

**模型**: `gemini-2.0-flash-preview-image-generation`

### 获取 API Key

1. 访问 [Google AI Studio](https://aistudio.google.com/)
2. 登录 Google 账号
3. 点击 "Get API key" → "Create API key"
4. 复制 API key 到 `.env` 的 `GOOGLE_API_KEY`

### 特点
- 支持文本生成和图像生成混合输出
- 支持参考图片输入（multimodal）
- 生成速度较快
- 免费层额度较充裕

### 限制
- 图像生成为 preview 功能，模型名称可能变更
- 部分地区需要 VPN

---

## OpenAI

**模型**: `gpt-image-1`

### 获取 API Key

1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 登录 → "API keys" → "Create new secret key"
3. 复制 API key 到 `.env` 的 `OPENAI_API_KEY`

### 特点
- 图像质量优秀
- 支持文本生成（`/images/generations`）和参考图编辑（`/images/edits`）
- 稳定的 API

### 宽高比映射

| 宽高比 | 像素尺寸 |
|--------|---------|
| 1:1 | 1024×1024 |
| 16:9 | 1536×1024 |
| 9:16 | 1024×1536 |
| 4:3 | 1536×1024 |

### 限制
- 按图片数量计费
- `gpt-image-1` 质量最高但价格较高

---

## Provider 选择建议

| 场景 | 推荐 Provider | 原因 |
|------|--------------|------|
| 日常迭代 | Google Gemini | 速度快、免费额度多 |
| 最终版图表 | OpenAI | 图像质量更高 |
| 有参考图 | Google Gemini | multimodal 支持更自然 |
| 需要精确尺寸 | OpenAI | 尺寸控制更精确 |

---

## 故障排查

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `status: 401` | API key 无效 | 检查 `.env` 中的 key 是否正确 |
| `status: 403` | 权限不足 | 确认 API key 有图像生成权限 |
| `status: 429` | 速率限制 | 等待后重试，或切换 provider |
| `status: 400` | 请求格式错误 | 检查 prompt 内容和参数 |
| 未找到 .env | 配置缺失 | 确认 `.env` 在项目根目录 |
