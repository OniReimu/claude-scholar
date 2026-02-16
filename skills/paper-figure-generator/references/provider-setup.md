# Provider 配置指南

AutoFigure-Edit 的后端服务配置。需要配置 LLM provider 和 SAM3 分割后端。

---

## 环境变量配置

在项目根目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# LLM provider（必需）
OPENROUTER_API_KEY=your-openrouter-api-key

# 可选：替代 LLM provider（AutoFigure-Edit 支持）
BIANXIE_API_KEY=your-bianxie-api-key

# SAM3 分割后端（推荐 Roboflow，免费）
ROBOFLOW_API_KEY=your-roboflow-api-key

# 可选：替代 SAM3 后端
# FAL_KEY=your-fal-key
```

优先级：`process.env` > 项目根目录 `.env`

---

## OpenRouter（LLM Provider，推荐）

**默认 LLM provider**，通过 OpenRouter 聚合多种模型。

### 获取 API Key

1. 访问 [OpenRouter](https://openrouter.ai/)
2. 注册/登录账号
3. 进入 Keys 页面 → 创建新 key
4. 复制 API key 到 `.env` 的 `OPENROUTER_API_KEY`

### 特点
- 聚合多种 LLM 模型（GPT-4o, Claude, Gemini 等）
- 按 token 计费，价格透明
- 统一 API 格式，兼容 OpenAI SDK
- 支持图像生成模型
- **注意**：这里提到的 Gemini / OpenAI 是**模型名称**（由 OpenRouter 路由到对应上游），并不意味着你需要 `GOOGLE_API_KEY` 或 `OPENAI_API_KEY`。使用本项目的 `paper-figure-generator` 只需要 `OPENROUTER_API_KEY`（以及 SAM3 后端的 `ROBOFLOW_API_KEY` 等）。

### 模型选择
- 默认模型由 AutoFigure-Edit 内部配置
- 可通过 `--image_model` 和 `--svg_model` 覆盖

---

## Roboflow（SAM3 Backend，推荐）

**默认 SAM3 分割后端**，免费 API 模式，无需本地安装 SAM3。

### 获取 API Key

1. 访问 [Roboflow](https://roboflow.com/)
2. 注册免费账号
3. 进入 Settings → API Keys
4. 复制 API key 到 `.env` 的 `ROBOFLOW_API_KEY`

### 特点
- 免费额度充裕
- 无需本地安装 SAM3 或 PyTorch
- API 模式，低资源占用
- 分割质量满足学术图表需求

### 使用
```bash
# 自动检测（脚本会根据环境变量选择后端）
bash scripts/generate.sh --method_file method.txt --output_dir output/

# 显式指定
bash scripts/generate.sh --sam_backend roboflow --method_file method.txt --output_dir output/
```

---

## fal.ai（替代 SAM3 Backend）

**替代 SAM3 后端**，提供更高精度的分割。

### 获取 API Key

1. 访问 [fal.ai](https://fal.ai/)
2. 注册账号
3. 进入 Dashboard → API Keys
4. 复制 API key 到 `.env` 的 `FAL_KEY`

### 特点
- 高精度分割
- 支持 `--sam_max_masks` 参数控制最大分割数
- 按调用次数计费

### 使用
```bash
bash scripts/generate.sh --sam_backend fal --method_file method.txt --output_dir output/
```

---

## 本地 SAM3（高级）

如果不想使用 API，可以本地安装 SAM3：

### 要求
- Python 3.12+
- PyTorch 2.7+
- 足够的 GPU 显存

### 安装
参考 [SAM3 官方仓库](https://github.com/facebookresearch/sam2) 进行本地安装。

### 使用
```bash
bash scripts/generate.sh --sam_backend local --method_file method.txt --output_dir output/
```

---

## Provider 选择建议

| 场景 | LLM Provider | SAM3 Backend | 原因 |
|------|-------------|-------------|------|
| 日常使用 | OpenRouter | Roboflow | 免费、无需本地安装 |
| 高精度分割 | OpenRouter | fal.ai | 分割质量更高 |
| 离线使用 | OpenRouter | local | 无需网络（LLM 仍需网络） |
| GPU 资源充裕 | OpenRouter | local | 无 API 调用限制 |

## 快速自检（推荐）

在生成前先跑一次 doctor，快速发现环境问题：

```bash
bash skills/paper-figure-generator/scripts/doctor.sh
```

---

## 故障排查

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `OPENROUTER_API_KEY not set` | 环境变量未配置 | 检查 `.env` 文件 |
| `AutoFigure-Edit not found` | 未安装 | 运行 `bash scripts/setup.sh` |
| SAM3 分割失败 | API key 无效或配额用尽 | 检查 API key，或切换后端 |
| 生成图片质量差 | 方法文本描述不清晰 | 改进 method.txt，添加更具体的组件和关系描述 |
| SVG 转 PDF 失败 | 缺少 cairosvg 或 inkscape | `uv pip install cairosvg` 或安装 Inkscape |
