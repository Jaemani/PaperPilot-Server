import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import rateLimit from "express-rate-limit";

dotenv.config();

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Rate limiter: 30 requests per minute per IP
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests. Please wait a moment." },
});
app.use("/analyze", limiter);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// 🛠️ Logging Middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  console.log("👉 Request Body:", JSON.stringify(req.body, null, 2));
  next();
});

// Profile context mapping for domain-specific term analysis
const getProfileContext = (profileId?: string): string => {
  if (!profileId) return "";

  // IEEE family: technical precision, formal engineering terminology
  if (profileId.includes("ieee") || profileId.includes("postech") || profileId.includes("kaist")) {
    return "\n\nJournal Context: IEEE/Technical — Prioritize technical precision and formal engineering terminology. Avoid conversational or colloquial terms.";
  }

  // Nature/Science family: accessible to broad scientific audience
  if (profileId.includes("nature") || profileId.includes("cell") || profileId.includes("science")) {
    return "\n\nJournal Context: Nature/Cell — Target broad scientific audience. Prefer clear, accessible language while maintaining scientific rigor. Avoid overly technical jargon when simpler terms exist.";
  }

  // ML conferences: modern terminology, accepts newer conventions
  if (profileId.includes("neurips") || profileId.includes("icml") || profileId.includes("aaai") ||
      profileId.includes("acl") || profileId.includes("emnlp")) {
    return "\n\nJournal Context: ML Conference — Modern AI/ML research community. Accepts contemporary terminology (e.g., 'we', 'our approach'). Prioritize clarity and common ML conventions.";
  }

  // ACM/Springer: balanced formal academic style
  if (profileId.includes("acm") || profileId.includes("springer")) {
    return "\n\nJournal Context: ACM/Springer — Standard academic formality. Balance technical precision with readability.";
  }

  return "";
};

// --- 1. Term Check API (Context-Aware with Profile) ---
app.post("/analyze/term", async (req: Request, res: Response) => {
  const { term, context, profileId } = req.body;
  const profileContext = getProfileContext(profileId);

  try {
    // @ts-ignore: Assuming new SDK types might not be fully updated in local environment yet
    const response = await openai.responses.create({
      model: "gpt-5",
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text: `You are an academic writing assistant. Analyze if the selected term is formal enough for top-tier research papers. Respond in JSON format.${profileContext}`
            }
          ]
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: `Term: "${term}"\nContext: "${context}"\n\nIs this term informal? If yes, provide 3 formal alternatives.
              If the term is nonsensical (gibberish), set suggestions to empty array.
              Return JSON: { "isInformal": boolean, "suggestions": string[], "reason": string }`
            }
          ]
        }
      ]
    });

    // @ts-ignore
    const content = response.output_text || "{}";
    const jsonResponse = JSON.parse(content);

    console.log(`✅ GPT Response (profile: ${profileId || "none"}):`, JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Failed to analyze term" });
  }
});

// --- 2. Batch Citation Analysis (Hybrid AI Strategy) ---
app.post("/analyze/citations-batch", async (req: Request, res: Response) => {
  const { candidates, profileId } = req.body;

  if (!Array.isArray(candidates) || candidates.length === 0) {
    return res.status(400).json({ error: "candidates array is required" });
  }

  if (candidates.length > 100) {
    return res.status(400).json({ error: "Maximum 100 candidates per batch" });
  }

  const profileContext = getProfileContext(profileId);

  try {
    // Build batch prompt
    const candidateList = candidates
      .map((c, i) => `[${i + 1}] ID: ${c.id}\n   Text: "${c.text}"\n   Context: "${c.context}"\n   Reason: ${c.reason}`)
      .join("\n\n");

    const prompt = `Analyze the following citation candidates for potential improvements. For each candidate, suggest:
- Whether to apply a range notation (e.g., [1], [2], [3] → [1-3])
- Whether to move the citation to a better position
- Whether the current style is acceptable as-is

Return a JSON array with one entry per candidate in the same order:
[
  {
    "id": "cite_ai_...",
    "action": "range" | "move" | "accept",
    "suggestion": "string (specific suggestion, or null if accept)",
    "confidence": "high" | "medium" | "low",
    "rationale": "brief explanation"
  }
]

Citation Candidates:
${candidateList}
${profileContext}`;

    // @ts-ignore
    const response = await openai.responses.create({
      model: "gpt-5",
      input: [
        {
          role: "system",
          content: [
            {
              type: "input_text",
              text: "You are an academic citation style advisor. Analyze citation placements and formats for research papers. Respond in JSON format."
            }
          ]
        },
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: prompt
            }
          ]
        }
      ]
    });

    // @ts-ignore
    const content = response.output_text || "[]";
    const jsonResponse = JSON.parse(content);

    console.log(`✅ Batch Citation Analysis (${candidates.length} candidates):`, JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Failed to analyze citations batch" });
  }
});

// --- 3. Format Parser API (Caption Parsing) ---
app.post("/analyze/format", async (req: Request, res: Response) => {
  const { rawCaption } = req.body;

  try {
    // @ts-ignore
    const response = await openai.responses.create({
      model: "gpt-5",
      input: [
        {
          role: "system",
          content: [
             { type: "input_text", text: "You are a structural parser. Extract components from a figure/table caption. Respond in JSON format." }
          ]
        },
        {
          role: "user",
          content: [
             { type: "input_text", text: `Caption: "${rawCaption}"\n\nExtract prefix, number, separator, and main content.
             Return JSON: { "prefix": string, "number": string, "separator": string, "content": string }` }
          ]
        }
      ]
    });

    // @ts-ignore
    const content = response.output_text || "{}";
    const jsonResponse = JSON.parse(content);

    console.log("✅ GPT Response:", JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Failed to parse format" });
  }
});

// --- 4. Cite Check API (Claim Classification) ---
app.post("/analyze/cite", async (req: Request, res: Response) => {
  const { sentence } = req.body;

  try {
    // @ts-ignore
    const response = await openai.responses.create({
      model: "gpt-5",
      input: [
        {
          role: "system",
          content: [
             { type: "input_text", text: "Classify if the sentence is a claim that needs a citation. Respond in JSON format." }
          ]
        },
        {
          role: "user",
          content: [
             { type: "input_text", text: `Sentence: "${sentence}"\n\nClassify as: "GENERAL" (fact), "OWN" (author's result), or "EXTERNAL" (external claim).
             Return JSON: { "type": "GENERAL" | "OWN" | "EXTERNAL", "reason": string }` }
          ]
        }
      ]
    });

    // @ts-ignore
    const content = response.output_text || "{}";
    const jsonResponse = JSON.parse(content);

    console.log("✅ GPT Response:", JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: "Failed to classify sentence" });
  }
});

app.listen(port, () => {
  console.log(`PaperPilot Backend running at http://localhost:${port}`);
});