// OpenAI 图像生成 Provider

import { readFileSync } from "node:fs";
import type { ProviderConfig } from "../types";

const DEFAULT_MODEL = "gpt-image-1";
const API_BASE = "https://api.openai.com/v1";

/** 宽高比到像素尺寸的映射 */
const AR_TO_SIZE: Record<string, string> = {
  "1:1": "1024x1024",
  "16:9": "1536x1024",
  "9:16": "1024x1536",
  "4:3": "1536x1024",
  "3:2": "1536x1024",
  "3:4": "1024x1536",
  "2:3": "1024x1536",
};

/** 通过 OpenAI API 生成图像 */
export async function generate(
  prompt: string,
  config: ProviderConfig,
  refImagePath?: string,
): Promise<{ imageData: string; mimeType: string }> {
  if (refImagePath) {
    return generateWithReference(prompt, config, refImagePath);
  }
  return generateFromText(prompt, config);
}

/** 纯文本生成 */
async function generateFromText(
  prompt: string,
  config: ProviderConfig,
): Promise<{ imageData: string; mimeType: string }> {
  const model = config.model || DEFAULT_MODEL;
  const size = AR_TO_SIZE[config.aspectRatio] || "1536x1024";
  const quality = config.quality === "high" ? "high" : "medium";

  const response = await fetch(`${API_BASE}/images/generations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model,
      prompt,
      n: 1,
      size,
      quality,
      output_format: "b64_json",
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenAI API 错误 status: ${response.status} - ${errorText}`,
    );
  }

  const data = (await response.json()) as {
    data: Array<{ b64_json: string }>;
  };

  if (!data.data?.[0]?.b64_json) {
    throw new Error("OpenAI API 响应中未找到图片数据");
  }

  return {
    imageData: data.data[0].b64_json,
    mimeType: "image/png",
  };
}

/** 带参考图片生成（使用 edits endpoint） */
async function generateWithReference(
  prompt: string,
  config: ProviderConfig,
  refImagePath: string,
): Promise<{ imageData: string; mimeType: string }> {
  const model = config.model || DEFAULT_MODEL;
  const size = AR_TO_SIZE[config.aspectRatio] || "1536x1024";

  const imageBuffer = readFileSync(refImagePath);
  const imageBlob = new Blob([imageBuffer], { type: "image/png" });

  const formData = new FormData();
  formData.append("model", model);
  formData.append("prompt", prompt);
  formData.append("image[]", imageBlob, "reference.png");
  formData.append("n", "1");
  formData.append("size", size);

  const response = await fetch(`${API_BASE}/images/edits`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenAI Edits API 错误 status: ${response.status} - ${errorText}`,
    );
  }

  const data = (await response.json()) as {
    data: Array<{ b64_json: string }>;
  };

  if (!data.data?.[0]?.b64_json) {
    throw new Error("OpenAI Edits API 响应中未找到图片数据");
  }

  return {
    imageData: data.data[0].b64_json,
    mimeType: "image/png",
  };
}
