const { GoogleGenAI } = require("@google/genai");

const MAX_CHARS = 300;

// ===============================
// Gemini API Key Pool（輪詢）
// ===============================
const GEMINI_KEYS = [
  process.env.GEMINI_API_KEY_1,
  process.env.GEMINI_API_KEY_2,
  process.env.GEMINI_API_KEY_3 // 可選
].filter(Boolean);

let geminiKeyIndex = 0;

function getNextGeminiClient() {
  if (GEMINI_KEYS.length === 0) {
    throw new Error("No Gemini API keys configured");
  }
  const key = GEMINI_KEYS[geminiKeyIndex];
  geminiKeyIndex = (geminiKeyIndex + 1) % GEMINI_KEYS.length;
  return new GoogleGenAI({ apiKey: key });
}

// ===============================
// 工具函式
// ===============================
function send(res, status, obj) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(obj));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function enforceMaxChars(text) {
  const s = String(text || "");
  if (s.length <= MAX_CHARS) return s;
  return s.slice(0, MAX_CHARS) + "…";
}

function extractFirstJsonObject(text) {
  const s = String(text || "");
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  return s.slice(start, end + 1);
}

// ===============================
// Gemini 模型映射
// ===============================
function mapGemini(uiModel) {
  const PRO = "gemini-3-pro-preview";
  const FLASH = "gemini-3-flash-preview";

  switch (uiModel) {
    case "gemini-3-pro":
    case "gemini-3-pro-standard":
    case "gemini-3-pro-intuition":
    case "gemini-3-thinking":
      return { pro: PRO, flash: FLASH };

    case "gemini-3-flash":
      return { pro: FLASH, flash: FLASH };

    case "gemini-3-pro-judge":
      return { pro: PRO, flash: FLASH };

    default:
      return { pro: PRO, flash: FLASH };
  }
}

// ===============================
// Gemini 呼叫（含 retry + fallback）
// ===============================
async function callGeminiWithFallback({ uiModel, prompt, temperature, forceJson }) {
  const { pro, flash } = mapGemini(uiModel);

  const basePrompt = forceJson
    ? prompt
    : (
        "【輸出限制】請用繁體中文回答，總長度 ≤ 300 字，用 2~5 點條列。\n\n" +
        prompt
      );

  const config = {};
  if (typeof temperature === "number") config.temperature = temperature;
  if (forceJson) config.responseMimeType = "application/json";

  let lastError = null;

  // --- 第 1 階段：嘗試 Pro（最多 2 次） ---
  for (let i = 0; i < 2; i++) {
    try {
      const ai = getNextGeminiClient();
      const resp = await ai.models.generateContent({
        model: pro,
        contents: basePrompt,
        config
      });
      return { model_used: pro, text: resp?.text ?? "" };
    } catch (e) {
      lastError = e;
      const msg = String(e.message || "");
      if (msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED")) {
        await sleep(1500);
        continue;
      }
      break;
    }
  }

  // --- 第 2 階段：自動降級 Flash ---
  try {
    const ai = getNextGeminiClient();
    const resp = await ai.models.generateContent({
      model: flash,
      contents: basePrompt,
      config
    });
    return { model_used: flash, text: resp?.text ?? "" };
  } catch (e) {
    // 最後保底
    if (forceJson) {
      return {
        model_used: flash,
        text: JSON.stringify({ type: "Type_C", reason: "Gemini 無法回應（限流）" })
      };
    }
    return {
      model_used: flash,
      text: "【Gemini 暫時無法回應，可能因流量或配額限制】"
    };
  }
}

// ===============================
// GPT-4o 裁判（OpenAI）
// ===============================
async function callOpenAIJudge(prompt) {
  const r = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o",
      input: prompt,
      temperature: 0
    })
  });

  const raw = await r.json();
  const text = raw.output?.[0]?.content?.[0]?.text ?? "";
  return { model_used: "gpt-4o", text };
}

function judgePromptWrapper(text) {
  return (
    "你是嚴格的科展評審裁判，只能輸出 JSON。\n" +
    '{ "type": "OK|Type_A|Type_B|Type_C|Type_D", "reason": "20字內理由" }\n\n' +
    text
  );
}

// ===============================
// Vercel Handler
// ===============================
module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.end();
  if (req.method !== "POST") return send(res, 405, { error: "Use POST" });

  const { prompt, model, temperature } = req.body || {};
  if (!prompt) return send(res, 400, { error: "Missing prompt" });

  try {
    // GPT-4o 裁判
    if (model === "gpt-4o-judge") {
      const jp = judgePromptWrapper(prompt);
      const out = await callOpenAIJudge(jp);
      const json = extractFirstJsonObject(out.text) || out.text;
      return send(res, 200, { result: json, real_model_used: out.model_used });
    }

    // Gemini 裁判
    if (model === "gemini-3-pro-judge") {
      const jp = judgePromptWrapper(prompt);
      const out = await callGeminiWithFallback({
        uiModel: model,
        prompt: jp,
        temperature: 0,
        forceJson: true
      });
      const json = extractFirstJsonObject(out.text)
        || JSON.stringify({ type: "Type_C", reason: "裁判回傳異常" });
      return send(res, 200, { result: json, real_model_used: out.model_used });
    }

    // Gemini 受試者（Subject）
    const out = await callGeminiWithFallback({
      uiModel: model,
      prompt,
      temperature,
      forceJson: false
    });

    return send(res, 200, {
      result: enforceMaxChars(out.text),
      real_model_used: out.model_used
    });

  } catch (e) {
    return send(res, 500, { error: String(e.message || e) });
  }
};
