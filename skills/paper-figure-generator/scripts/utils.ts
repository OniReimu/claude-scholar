// 共享工具函数：环境变量加载、参数解析、重试机制

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import type { CLIArgs, Provider } from "./types";

// ==================== 环境变量 ====================

/** 向上查找项目根目录的 .env 文件并加载 */
export function loadEnv(): void {
  const envPath = findFileUpward(".env");
  if (!envPath) return;

  const content = readFileSync(envPath, "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) continue;

    const key = trimmed.slice(0, eqIndex).trim();
    const value = trimmed.slice(eqIndex + 1).trim().replace(/^["']|["']$/g, "");

    // process.env 优先，不覆盖已有值
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

/** 从当前目录向上查找文件 */
function findFileUpward(filename: string): string | null {
  let dir = process.cwd();
  while (true) {
    const candidate = join(dir, filename);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

// ==================== 参数解析 ====================

/** 解析命令行参数（无外部依赖） */
export function parseArgs(argv: string[]): CLIArgs {
  const args: CLIArgs = {};
  const rawArgs = argv.slice(2);

  for (let i = 0; i < rawArgs.length; i++) {
    const arg = rawArgs[i];
    const next = rawArgs[i + 1];

    switch (arg) {
      case "--prompt":
        args.prompt = next;
        i++;
        break;
      case "--promptfiles": {
        const files = next?.split(",").map((f) => f.trim()) ?? [];
        args.promptfiles = [...(args.promptfiles ?? []), ...files];
        i++;
        break;
      }
      case "--provider":
        args.provider = next as Provider;
        i++;
        break;
      case "--model":
        args.model = next;
        i++;
        break;
      case "--ar":
        args.ar = next;
        i++;
        break;
      case "--quality":
        args.quality = next as "normal" | "high";
        i++;
        break;
      case "--ref":
        args.ref = next;
        i++;
        break;
      case "--output":
        args.output = next;
        i++;
        break;
    }
  }

  return args;
}

// ==================== Prompt 处理 ====================

/** 读取并拼接 prompt 文件内容，或使用内联 prompt */
export function resolvePrompt(args: CLIArgs): string {
  const parts: string[] = [];

  if (args.promptfiles?.length) {
    for (const file of args.promptfiles) {
      const filePath = resolve(file);
      if (!existsSync(filePath)) {
        throw new Error(`Prompt 文件不存在: ${filePath}`);
      }
      parts.push(readFileSync(filePath, "utf-8").trim());
    }
  }

  if (args.prompt) {
    parts.push(args.prompt);
  }

  const combined = parts.join("\n\n");
  if (!combined) {
    throw new Error("未提供 prompt：使用 --prompt 或 --promptfiles");
  }

  return combined;
}

// ==================== Provider 检测 ====================

/** 根据环境变量自动检测可用的 provider（Google 优先） */
export function detectProvider(): Provider {
  const defaultProvider = process.env.DEFAULT_PROVIDER as Provider | undefined;
  if (defaultProvider && isValidProvider(defaultProvider)) {
    return defaultProvider;
  }

  if (process.env.GOOGLE_API_KEY) return "google";
  if (process.env.OPENAI_API_KEY) return "openai";

  throw new Error(
    "未找到 API key。请在 .env 文件中设置 GOOGLE_API_KEY 或 OPENAI_API_KEY",
  );
}

function isValidProvider(p: string): p is Provider {
  return p === "google" || p === "openai";
}

// ==================== 图片保存 ====================

/** 将 base64 图片数据保存为 PNG 文件 */
export function saveImage(base64Data: string, outputPath: string): string {
  const absPath = resolve(outputPath);
  const dir = dirname(absPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const buffer = Buffer.from(base64Data, "base64");
  writeFileSync(absPath, buffer);
  return absPath;
}

// ==================== 重试机制 ====================

/** 不可重试的 HTTP 状态码 */
const NON_RETRYABLE_CODES = new Set([400, 401, 403, 404, 422]);

/** 带指数退避的重试函数 */
export async function retry<T>(
  fn: () => Promise<T>,
  maxAttempts = 3,
  baseDelay = 1000,
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));

      // 检查是否为不可重试错误
      const statusMatch = lastError.message.match(/status[:\s]*(\d{3})/i);
      if (statusMatch && NON_RETRYABLE_CODES.has(Number(statusMatch[1]))) {
        throw lastError;
      }

      if (attempt < maxAttempts) {
        const delay = baseDelay * Math.pow(2, attempt - 1);
        console.error(
          `尝试 ${attempt}/${maxAttempts} 失败，${delay}ms 后重试...`,
        );
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  throw lastError;
}

// ==================== 用法提示 ====================

export function printUsage(): void {
  console.log(`
Paper Figure Generator - 学术论文图表生成工具

用法:
  npx -y bun scripts/main.ts [选项]

选项:
  --prompt <text>        内联 prompt 文本
  --promptfiles <files>  prompt 文件路径（逗号分隔）
  --provider <name>      指定 provider: google | openai
  --model <name>         模型名称
  --ar <ratio>           宽高比: 16:9 | 1:1 | 4:3 | 3:2（默认 16:9）
  --quality <level>      图片质量: normal | high（默认 normal）
  --ref <path>           参考图片路径
  --output <path>        输出文件路径（默认 output.png）

示例:
  # 从 prompt 文件生成
  npx -y bun scripts/main.ts --promptfiles prompt.md --output figure.png

  # 内联 prompt + 参考图
  npx -y bun scripts/main.ts --prompt "system overview diagram" --ref ref.png

  # 指定 provider 和宽高比
  npx -y bun scripts/main.ts --promptfiles prompt.md --provider openai --ar 4:3

环境变量:
  GOOGLE_API_KEY     Google Gemini API key
  OPENAI_API_KEY     OpenAI API key
  DEFAULT_PROVIDER   默认 provider（google | openai）
`);
}
