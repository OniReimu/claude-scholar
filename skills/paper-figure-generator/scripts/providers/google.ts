// Google Gemini 图像生成 Provider

import { readFileSync } from "node:fs";
import type { ProviderConfig } from "../types";

const DEFAULT_MODEL = "gemini-3-pro-image-preview";

/** 通过 Gemini API 生成图像 */
export async function generate(
  prompt: string,
  config: ProviderConfig,
  refImagePath?: string,
): Promise<{ imageData: string; mimeType: string }> {
  const model = config.model || DEFAULT_MODEL;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;

  const contents = buildContents(prompt, refImagePath);

  const body = {
    contents,
    generationConfig: {
      responseModalities: ["TEXT", "IMAGE"],
    },
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": config.apiKey,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Gemini API 错误 status: ${response.status} - ${errorText}`,
    );
  }

  const data = await response.json();
  return extractImage(data);
}

/** 构建请求内容（支持参考图） */
function buildContents(
  prompt: string,
  refImagePath?: string,
): Array<{ role: string; parts: Array<Record<string, unknown>> }> {
  const parts: Array<Record<string, unknown>> = [];

  // 添加参考图片
  if (refImagePath) {
    const imageData = readFileSync(refImagePath).toString("base64");
    const mimeType = refImagePath.endsWith(".png") ? "image/png" : "image/jpeg";
    parts.push({
      inlineData: {
        mimeType,
        data: imageData,
      },
    });
    parts.push({
      text: `Reference the style and layout of the image above. ${prompt}`,
    });
  } else {
    parts.push({ text: prompt });
  }

  return [{ role: "user", parts }];
}

/** 从 API 响应中提取图片数据 */
function extractImage(
  data: Record<string, unknown>,
): { imageData: string; mimeType: string } {
  const candidates = data.candidates as Array<{
    content: { parts: Array<{ inlineData?: { data: string; mimeType: string } }> };
  }>;

  if (!candidates?.length) {
    throw new Error("Gemini API 未返回候选结果");
  }

  const parts = candidates[0].content.parts;
  for (const part of parts) {
    if (part.inlineData?.data) {
      return {
        imageData: part.inlineData.data,
        mimeType: part.inlineData.mimeType || "image/png",
      };
    }
  }

  throw new Error("Gemini API 响应中未找到图片数据");
}
