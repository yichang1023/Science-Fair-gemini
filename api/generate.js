const { GoogleGenAI } = require("@google/genai");

const MAX_CHARS = 300; // ✅ 你要的回應字數上限（繁中約 300 字）

function send(res, status, obj) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(obj));
}

function mustEnv(name) {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

// ✅ 強制截斷，避免回傳過長（保證 <= MAX_CHARS）
function enforceMaxChars(text, maxChars = MAX_CHARS) {
  const s = String(text || "");
  if (s.length <= maxChars) return s;
  return s.slice(0, maxChars) + "…";
}

// 讓前端 JSON.parse() 穩定：抽出第一個 {...}
function extractFirstJsonObject(text) {
  const s = String(text || "");
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  return s.slice(start, end + 1);
}

// 你的 UI model value → 真實 Gemini 3 模型 + config
function mapGemini(uiModel) {
  const PRO = "gemini-3-pro-preview";
  const FLASH = "gemini-3-flash-preview";

  switch (uiModel) {
    case "gemini-3-pro-standard":
      return { realModel: PRO, config: {} };

    case "gemini-3-pro-safe":
      return {
        realModel: PRO,
        config: {
          safetySettings: [
            { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_LOW_AND_ABOVE" },
            { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_LOW_AND_ABOVE" },
            { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_LOW_AND_ABOVE" },
            { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_LOW_AND_ABOVE" }
          ]
        }
      };

    case "gemini-3-pro":
      return { realModel: PRO, config: {} };

    case "gemini-3-flash":
      return {
        realModel: FLASH,
        config: { thinkingConfig: { thinkingLevel: "low" } }
      };

    case "gemini-3-pro-intuition":
      return {
        realModel: PRO,
        config: { thinkingConfig: { thinkingLevel: "low" } }
      };

    case "gemini-3-thinking":
      return {
        realModel: PRO,
        config: { thinkingConfig: { thinkingLevel: "high" } }
      };

    case "gemini-3-pro-judge":
      return { realModel: PRO, config: {} };

    default:
      return { realModel: PRO, config: {} };
  }
}

async function callGemini({ uiModel, prompt, temperature, forceJson }) {
  const apiKey = mustEnv("GEMINI_API_KEY");
  const ai = new GoogleGenAI({ apiKey });

  const { realModel, config } = mapGemini(uiModel);

  const finalConfig = { ...config };
  if (typeof temperature === "number" && !Number.isNaN(temperature)) {
    finalConfig.temperature = temperature;
  }
  if (forceJson) {
    finalConfig.responseMimeType = "application/json";
  }

  // ✅ 只對「非 JSON 模式（subject）」加字數規範，避免裁判 JSON 被影響
  const effectivePrompt = forceJson
    ? prompt
    : (
        "【輸出規格】請用繁體中文回答，總長度必須 ≤ 300 字（含標點）。" +
        "用 2~5 點條列，每點一句話。若不確定，請明確說不確定。\n\n" +
        prompt
      );

  const resp = await ai.models.generateContent({
    model: realModel,
    contents: effectivePrompt,
    config: finalConfig
  });

  return {
    real_model_used: realModel,
    text: resp?.text ?? ""
  };
}

async function callOpenAIJson({ prompt }) {
  const apiKey = mustEnv("OPENAI_API_KEY");

  const r = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
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
  return { real_model_used: "gpt-4o", text };
}

function judgePromptWrapper(userText) {
  return (
    "你是嚴格的科展評審裁判。只能輸出 JSON（不可多字）。\n" +
    'JSON 格式：{ "type": "OK|Type_A|Type_B|Type_C|Type_D", "reason": "20字內理由" }\n' +
    "類型：A=瞎掰/事實錯誤，B=錯置/張冠李戴，C=推理矛盾，D=拒答/過度保守，OK=正確。\n\n" +
    "待評測內容：\n" + String(userText || "")
  );
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.end();
  if (req.method !== "POST") return send(res, 405, { error: "Use POST" });

  let body = {};
  try {
    body = typeof req.body === "object" ? req.body : JSON.parse(req.body || "{}");
  } catch {
    return send(res, 400, { error: "Invalid JSON" });
  }

  const prompt = String(body.prompt || "");
  const model = String(body.model || "gemini-3-pro-standard");
  const temperature = typeof body.temperature === "number" ? body.temperature : undefined;

  if (!prompt) return send(res, 400, { error: "Missing prompt" });

  try {
    // ① GPT-4o 裁判（JSON 不截斷，避免壞 JSON）
    if (model === "gpt-4o-judge") {
      const jp = judgePromptWrapper(prompt);
      const out = await callOpenAIJson({ prompt: jp });
      const jsonStr = extractFirstJsonObject(out.text) || out.text;
      return send(res, 200, { result: jsonStr, real_model_used: out.real_model_used });
    }

    // ② Gemini 裁判（JSON 不截斷，避免壞 JSON）
    if (model === "gemini-3-pro-judge") {
      const jp = judgePromptWrapper(prompt);
      const out = await callGemini({ uiModel: model, prompt: jp, temperature: 0, forceJson: true });
      const jsonStr = extractFirstJsonObject(out.text);
      const safe = jsonStr || JSON.stringify({ type: "Type_C", reason: "裁判輸出非JSON" });
      return send(res, 200, { result: safe, real_model_used: out.real_model_used });
    }

    // ③ Gemini 選手（真正 Gemini 3）—— ✅ 強制回傳 ≤ 300 字
    const out = await callGemini({ uiModel: model, prompt, temperature, forceJson: false });
    const clipped = enforceMaxChars(out.text, MAX_CHARS);
    return send(res, 200, { result: clipped, real_model_used: out.real_model_used });

  } catch (e) {
    return send(res, 500, { error: String(e?.message || e) });
  }
};
