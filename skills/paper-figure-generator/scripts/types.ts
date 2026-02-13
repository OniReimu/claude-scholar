// 图像生成 Provider 和 CLI 参数类型定义

export type Provider = "google" | "openai";

export interface CLIArgs {
  /** 内联 prompt 文本 */
  prompt?: string;
  /** prompt 文件路径（逗号分隔或多次指定） */
  promptfiles?: string[];
  /** 指定 provider */
  provider?: Provider;
  /** 模型名称 */
  model?: string;
  /** 宽高比，如 "16:9", "1:1", "4:3" */
  ar?: string;
  /** 图片质量 */
  quality?: "normal" | "high";
  /** 参考图片路径 */
  ref?: string;
  /** 输出文件路径 */
  output?: string;
}

export interface GenerationResult {
  success: boolean;
  imagePath?: string;
  error?: string;
  provider: Provider;
  model: string;
}

export interface ProviderConfig {
  apiKey: string;
  model: string;
  aspectRatio: string;
  quality: string;
}

/** Provider 生成函数签名 */
export type GenerateFunction = (
  prompt: string,
  config: ProviderConfig,
  refImagePath?: string,
) => Promise<{ imageData: string; mimeType: string }>;
