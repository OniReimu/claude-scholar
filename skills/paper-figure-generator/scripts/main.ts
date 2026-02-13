#!/usr/bin/env bun
// Paper Figure Generator - 学术论文图表生成入口

import type { GenerateFunction, ProviderConfig } from "./types";
import {
  loadEnv,
  parseArgs,
  resolvePrompt,
  detectProvider,
  saveImage,
  retry,
  printUsage,
} from "./utils";

async function main(): Promise<void> {
  const args = parseArgs(process.argv);

  // 无参数时显示用法
  if (!args.prompt && !args.promptfiles?.length) {
    printUsage();
    process.exit(0);
  }

  // 1. 加载环境变量
  loadEnv();

  // 2. 解析 prompt
  const prompt = resolvePrompt(args);
  console.log(`📝 Prompt 长度: ${prompt.length} 字符`);

  // 3. 检测 provider
  const provider = args.provider ?? detectProvider();
  console.log(`🔌 Provider: ${provider}`);

  // 4. 加载 provider 模块
  const generateFn = await loadProvider(provider);

  // 5. 构建配置
  const config: ProviderConfig = {
    apiKey: getApiKey(provider),
    model: args.model ?? "",
    aspectRatio: args.ar ?? "16:9",
    quality: args.quality ?? "normal",
  };

  // 6. 生成图片（带重试）
  console.log("🎨 正在生成图片...");
  const result = await retry(
    () => generateFn(prompt, config, args.ref),
    3,
    2000,
  );

  // 7. 保存图片
  const outputPath = args.output ?? "output.png";
  const savedPath = saveImage(result.imageData, outputPath);
  console.log(`✅ 图片已保存: ${savedPath}`);
}

/** 动态加载 provider 模块 */
async function loadProvider(provider: string): Promise<GenerateFunction> {
  switch (provider) {
    case "google": {
      const mod = await import("./providers/google");
      return mod.generate;
    }
    case "openai": {
      const mod = await import("./providers/openai");
      return mod.generate;
    }
    default:
      throw new Error(`不支持的 provider: ${provider}`);
  }
}

/** 获取 provider 对应的 API key */
function getApiKey(provider: string): string {
  const keyMap: Record<string, string> = {
    google: "GOOGLE_API_KEY",
    openai: "OPENAI_API_KEY",
  };

  const envName = keyMap[provider];
  if (!envName) throw new Error(`不支持的 provider: ${provider}`);

  const key = process.env[envName];
  if (!key) throw new Error(`缺少 ${envName}，请在 .env 文件中配置`);

  return key;
}

main().catch((err) => {
  console.error(`❌ 生成失败: ${err.message}`);
  process.exit(1);
});
