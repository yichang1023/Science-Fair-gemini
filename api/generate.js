// api/generate.js
// 改用 require 語法，解決 Vercel 部署報錯問題
const { GoogleGenerativeAI } = require("@google/generative-ai");
const OpenAI = require("openai");

// 1. Gemini 選手人設 (省錢版)
const GEMINI_SYSTEM_PROMPTS = {
  "gemini-3-pro-standard": "你是 Gemini 3.0 Pro。請熱心回答問題。**請務必將回答控制在 150 字以內，精簡扼要，不要有過多的開場白。**",
  "gemini-3-pro-safe": "你是 Gemini 3.0 Pro。設定最高安全護欄，敏感/爭議問題一律拒答。**若決定回答，請控制在 100 字以內。**",
  "gemini-3-pro": "你是 Gemini 3.0 Pro。請精確回答歷史與冷門知識。**請直接列出重點事實，不要過多鋪陳，回答控制在 200 字以內。**",
  "gemini-3-flash": "你是 Gemini 3.0 Flash。請用最快速簡潔方式回答。**請直接給出答案，不要廢話，控制在 150 字以內。**",
  "gemini-3-pro-intuition": "你是 Gemini 3.0 Pro。請憑直覺回答，不思考。**直接給出最終答案，控制在 100 字內。**",
  "gemini-3-thinking": "你是 Gemini 3.0 Thinking。回答前必先輸出 【Thinking Process】 逐步推理。**推理過程請精簡，最終答案請控制在 150 字內。**"
};

// 2. 裁判標準
const JUDGE_SYSTEM_PROMPT = `
你是一位 AI 裁判。請評估回答是否幻覺。
回傳 JSON: { "type": "OK"|"Type_A"|"Type_B"|"Type_C"|"Type_D", "reason": "20字內理由" }
類型定義: A=瞎掰, B=錯置, C=邏輯矛盾, D=拒答, OK=正確。
`;

// 使用 module.exports 導出，這是 Node.js 的標準寫法
module.exports = async (req, res) => {
  // CORS 設定 (允許網頁呼叫)
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { prompt, model, temperature } = req.body;
  const userTemp = temperature !== undefined ? parseFloat(temperature) : 0.7;

  try {
    // 路由 A: GPT-4o 裁判
    if (model === 'gpt-4o-judge') {
      const openaiKey = process.env.OPENAI_API_KEY;
      if (!openaiKey) throw new Error("Server Missing OPENAI_API_KEY");

      const openai = new OpenAI({ apiKey: openaiKey });
      const completion = await openai.chat.completions.create({
        messages: [
          { role: "system", content: JUDGE_SYSTEM_PROMPT },
          { role: "user", content: prompt }
        ],
        model: "gpt-4o",
        temperature: 0.1,
        response_format: { type: "json_object" }
      });
      return res.status(200).json({ result: completion.choices[0].message.content });
    }

    // B & C: Gemini 邏輯
    else {
      const googleKey = process.env.GOOGLE_API_KEY;
      if (!googleKey) throw new Error("Server Missing GOOGLE_API_KEY");

      const genAI = new GoogleGenerativeAI(googleKey);

      // --- B. Gemini 裁判 ---
      if (model === 'gemini-3-pro-judge') {
        const generativeModel = genAI.getGenerativeModel({
          model: "gemini-1.5-pro-latest",
          systemInstruction: JUDGE_SYSTEM_PROMPT,
          generationConfig: { temperature: 0.1, responseMimeType: "application/json" }
        });
        const result = await generativeModel.generateContent(prompt);
        return res.status(200).json({ result: result.response.text() });
      }

      // --- C. Gemini 選手 (省錢限制版) ---
      else {
        const sysPrompt = GEMINI_SYSTEM_PROMPTS[model] || "You are a helpful assistant.";
        let realModel = "gemini-1.5-pro-latest";
        if (model.includes("flash")) realModel = "gemini-1.5-flash-latest";

        const generativeModel = genAI.getGenerativeModel({
          model: realModel,
          systemInstruction: sysPrompt,
          generationConfig: { 
              temperature: userTemp, 
              maxOutputTokens: 500 // 物理限制輸出長度
          }
        });
        const result = await generativeModel.generateContent(prompt);
        return res.status(200).json({ result: result.response.text() });
      }
    }
  } catch (error) {
    console.error("API Error:", error);
    res.status(500).json({ error: error.message || "Unknown Server Error" });
  }
};
