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
  timeout: 30000, // 30 second timeout
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

// Helper: Safe JSON parsing with fallback
const parseJSONResponse = (text: string, fallback: any = {}): any => {
  try {
    // Try to extract JSON from markdown code blocks if present
    const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/) || text.match(/```\s*([\s\S]*?)\s*```/);
    const cleanText = jsonMatch ? jsonMatch[1] : text;
    return JSON.parse(cleanText.trim());
  } catch (error) {
    console.error("❌ JSON parsing failed:", error);
    console.error("Raw text:", text);
    return fallback;
  }
};

// --- 1. Term Check API (Context-Aware with Profile) ---
app.post("/analyze/term", async (req: Request, res: Response) => {
  const { term, context, profileId } = req.body;
  const profileContext = getProfileContext(profileId);

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: `You are an academic writing assistant. Analyze if the selected term is formal enough for top-tier research papers. Respond ONLY with valid JSON format.${profileContext}`
        },
        {
          role: "user",
          content: `Term: "${term}"\nContext: "${context}"\n\nIs this term informal? If yes, provide 3 formal alternatives.
If the term is nonsensical (gibberish), set suggestions to empty array.
Return JSON: { "isInformal": boolean, "suggestions": string[], "reason": string }`
        }
      ],
      temperature: 0.3,
      max_tokens: 500,
    });

    const responseText = completion.choices[0]?.message?.content || "{}";
    const jsonResponse = parseJSONResponse(responseText, {
      isInformal: false,
      suggestions: [],
      reason: "Unable to analyze term"
    });

    console.log(`✅ GPT Response (profile: ${profileId || "none"}):`, JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error: any) {
    console.error("❌ Error:", error.message);
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      res.status(504).json({ error: "Request timeout. Please try again." });
    } else if (error.status === 429) {
      res.status(429).json({ error: "Rate limit exceeded. Please wait a moment." });
    } else {
      res.status(500).json({ error: "Failed to analyze term", details: error.message });
    }
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

    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "You are an academic citation style advisor. Analyze citation placements and formats for research papers. Respond ONLY with valid JSON format."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.2,
      max_tokens: 2000,
    });

    const responseText = completion.choices[0]?.message?.content || "[]";
    const jsonResponse = parseJSONResponse(responseText, []);

    console.log(`✅ Batch Citation Analysis (${candidates.length} candidates):`, JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error: any) {
    console.error("❌ Error:", error.message);
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      res.status(504).json({ error: "Request timeout. Please try again." });
    } else if (error.status === 429) {
      res.status(429).json({ error: "Rate limit exceeded. Please wait a moment." });
    } else {
      res.status(500).json({ error: "Failed to analyze citations batch", details: error.message });
    }
  }
});

// --- 3. Format Parser API (Caption Parsing) ---
app.post("/analyze/format", async (req: Request, res: Response) => {
  const { rawCaption } = req.body;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "You are a structural parser. Extract components from a figure/table caption. Respond ONLY with valid JSON format."
        },
        {
          role: "user",
          content: `Caption: "${rawCaption}"\n\nExtract prefix, number, separator, and main content.
Return JSON: { "prefix": string, "number": string, "separator": string, "content": string }`
        }
      ],
      temperature: 0.1,
      max_tokens: 300,
    });

    const responseText = completion.choices[0]?.message?.content || "{}";
    const jsonResponse = parseJSONResponse(responseText, {
      prefix: "",
      number: "",
      separator: "",
      content: rawCaption
    });

    console.log("✅ GPT Response:", JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error: any) {
    console.error("❌ Error:", error.message);
    res.status(500).json({ error: "Failed to parse format", details: error.message });
  }
});

// --- 4. Reference Formatting API (DOI/Title → Journal Style) ---
app.post("/analyze/format-reference", async (req: Request, res: Response) => {
  const { input, style, profileId } = req.body;

  if (!input || typeof input !== 'string') {
    return res.status(400).json({ error: "input (DOI or title) is required" });
  }

  // Detect input type
  const isDOI = /^10\.\d{4,9}\/\S+$/i.test(input.trim());
  const isArXiv = /arxiv:\s*\d{4}\.\d{4,5}/i.test(input);

  try {
    // Step 1: Fetch metadata using web search
    let searchQuery = "";
    if (isDOI) {
      searchQuery = `doi:${input} citation metadata`;
    } else if (isArXiv) {
      searchQuery = `${input} citation`;
    } else {
      searchQuery = `"${input}" academic paper citation`;
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: `You are a citation formatting assistant. Given a DOI, arXiv ID, or paper title, format it according to the specified style (IEEE, Nature, APA, etc.). Use web search to find accurate metadata. Respond ONLY with valid JSON format.`
        },
        {
          role: "user",
          content: `Input: ${input}
Style: ${style || "IEEE"}
Profile: ${profileId || "generic"}

Search for this paper's metadata and format it according to ${style || "IEEE"} citation style.

Return JSON:
{
  "formatted": "full formatted citation string",
  "authors": ["Author 1", "Author 2"],
  "title": "Paper Title",
  "venue": "Conference/Journal Name",
  "year": 2024,
  "doi": "10.xxxx/xxxxx",
  "confidence": "high" | "medium" | "low"
}`
        }
      ],
      temperature: 0.2,
      max_tokens: 800,
    });

    const responseText = completion.choices[0]?.message?.content || "{}";
    const jsonResponse = parseJSONResponse(responseText, {
      formatted: input,
      authors: [],
      title: input,
      venue: "",
      year: 0,
      doi: "",
      confidence: "low"
    });

    console.log(`✅ Reference formatted:`, JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error: any) {
    console.error("❌ Error:", error.message);
    res.status(500).json({ error: "Failed to format reference", details: error.message });
  }
});

// --- 5. Cite Check API (Claim Classification) ---
app.post("/analyze/cite", async (req: Request, res: Response) => {
  const { sentence } = req.body;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "Classify if the sentence is a claim that needs a citation. Respond ONLY with valid JSON format."
        },
        {
          role: "user",
          content: `Sentence: "${sentence}"\n\nClassify as: "GENERAL" (fact), "OWN" (author's result), or "EXTERNAL" (external claim).
Return JSON: { "type": "GENERAL" | "OWN" | "EXTERNAL", "reason": string }`
        }
      ],
      temperature: 0.2,
      max_tokens: 300,
    });

    const responseText = completion.choices[0]?.message?.content || "{}";
    const jsonResponse = parseJSONResponse(responseText, {
      type: "GENERAL",
      reason: "Unable to classify"
    });

    console.log("✅ GPT Response:", JSON.stringify(jsonResponse, null, 2));
    res.json(jsonResponse);
  } catch (error: any) {
    console.error("❌ Error:", error.message);
    res.status(500).json({ error: "Failed to classify sentence", details: error.message });
  }
});

// --- 6. Paper Review API (Multi-Agent Reviewer System) ---
app.post("/analyze/review-paper", async (req: Request, res: Response) => {
  const { sections, venue, profileId } = req.body;

  if (!sections || typeof sections !== 'object') {
    return res.status(400).json({ error: "sections object is required" });
  }

  const { abstract, introduction, method, results, discussion } = sections;

  if (!abstract || !introduction || !method || !results) {
    return res.status(400).json({
      error: "Required sections: abstract, introduction, method, results"
    });
  }

  const profileContext = getProfileContext(profileId);
  const venueContext = venue ? `\nTarget Venue: ${venue}` : "";

  try {
    // Stage 1: Parallel section analysis (3 reviewers)
    const reviewerPromises = [
      // Reviewer A: Theorist (Novelty & Formalism)
      openai.chat.completions.create({
        model: "gpt-5-mini",
        messages: [
          {
            role: "system",
            content: `You are Reviewer A, a theorist evaluating novelty and technical soundness. Focus on: (1) Contribution clarity, (2) Novelty assessment, (3) Technical rigor, (4) Problem formulation.${profileContext}${venueContext}\n\nRespond ONLY with valid JSON.`
          },
          {
            role: "user",
            content: `Review this paper:

Abstract: ${abstract}

Introduction: ${introduction.substring(0, 2000)}

Method: ${method.substring(0, 2000)}

Provide JSON:
{
  "score": 0-10,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "comment": "2-3 sentence summary"
}`
          }
        ],
        temperature: 0.3,
        max_tokens: 600
      }),

      // Reviewer B: Experimentalist (Empirical Rigor)
      openai.chat.completions.create({
        model: "gpt-5-mini",
        messages: [
          {
            role: "system",
            content: `You are Reviewer B, an experimentalist evaluating empirical rigor. Focus on: (1) Experimental design, (2) Baseline comparisons, (3) Statistical validity, (4) Reproducibility.${profileContext}${venueContext}\n\nRespond ONLY with valid JSON.`
          },
          {
            role: "user",
            content: `Review this paper:

Method: ${method.substring(0, 2000)}

Results: ${results.substring(0, 2000)}

Provide JSON:
{
  "score": 0-10,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "comment": "2-3 sentence summary"
}`
          }
        ],
        temperature: 0.3,
        max_tokens: 600
      }),

      // Reviewer C: Impact Assessor (Significance)
      openai.chat.completions.create({
        model: "gpt-5-mini",
        messages: [
          {
            role: "system",
            content: `You are Reviewer C, assessing impact and significance. Focus on: (1) Problem importance, (2) Practical applicability, (3) Community value, (4) Long-term impact.${profileContext}${venueContext}\n\nRespond ONLY with valid JSON.`
          },
          {
            role: "user",
            content: `Review this paper:

Abstract: ${abstract}

Introduction: ${introduction.substring(0, 1500)}

${discussion ? `Discussion: ${discussion.substring(0, 1500)}` : ""}

Provide JSON:
{
  "score": 0-10,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "comment": "2-3 sentence summary"
}`
          }
        ],
        temperature: 0.3,
        max_tokens: 600
      })
    ];

    const [revA, revB, revC] = await Promise.all(reviewerPromises);

    const reviewerA = parseJSONResponse(revA.choices[0]?.message?.content || "{}", {
      score: 5, strengths: [], weaknesses: [], comment: ""
    });
    const reviewerB = parseJSONResponse(revB.choices[0]?.message?.content || "{}", {
      score: 5, strengths: [], weaknesses: [], comment: ""
    });
    const reviewerC = parseJSONResponse(revC.choices[0]?.message?.content || "{}", {
      score: 5, strengths: [], weaknesses: [], comment: ""
    });

    // Calculate overall score
    const overallScore = ((reviewerA.score + reviewerB.score + reviewerC.score) / 3).toFixed(1);
    const acceptProbability = Math.round(Math.min(100, Math.max(0, (parseFloat(overallScore) - 3) * 16.67)));

    // Determine recommendation
    let recommendation = "reject";
    if (parseFloat(overallScore) >= 8) recommendation = "strong_accept";
    else if (parseFloat(overallScore) >= 7) recommendation = "weak_accept";
    else if (parseFloat(overallScore) >= 6) recommendation = "borderline_accept";
    else if (parseFloat(overallScore) >= 5) recommendation = "borderline_reject";
    else if (parseFloat(overallScore) >= 4) recommendation = "weak_reject";

    const response = {
      overallScore: parseFloat(overallScore),
      acceptProbability,
      recommendation,
      reviewerScores: [
        {
          persona: "Theorist",
          focus: "novelty_and_formalism",
          score: reviewerA.score,
          strengths: reviewerA.strengths,
          weaknesses: reviewerA.weaknesses,
          detailedComment: reviewerA.comment
        },
        {
          persona: "Experimentalist",
          focus: "empirical_rigor",
          score: reviewerB.score,
          strengths: reviewerB.strengths,
          weaknesses: reviewerB.weaknesses,
          detailedComment: reviewerB.comment
        },
        {
          persona: "Impact_Assessor",
          focus: "significance_and_impact",
          score: reviewerC.score,
          strengths: reviewerC.strengths,
          weaknesses: reviewerC.weaknesses,
          detailedComment: reviewerC.comment
        }
      ],
      criticalIssues: [
        ...reviewerA.weaknesses.map((w: string, i: number) => ({
          id: `issue_a${i}`,
          severity: "medium",
          category: "novelty",
          issue: w
        })),
        ...reviewerB.weaknesses.map((w: string, i: number) => ({
          id: `issue_b${i}`,
          severity: "high",
          category: "experiment",
          issue: w
        }))
      ].slice(0, 5) // Top 5 issues
    };

    console.log(`✅ Paper Review Complete (score: ${overallScore}/10)`);
    res.json(response);
  } catch (error: any) {
    console.error("❌ Error:", error.message);
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      res.status(504).json({ error: "Request timeout. Paper review takes 1-2 minutes." });
    } else {
      res.status(500).json({ error: "Failed to review paper", details: error.message });
    }
  }
});

// Health check endpoint
app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

app.listen(port, () => {
  console.log(`🚀 PaperPilot Server v1.4.0 running at http://localhost:${port}`);
  console.log(`📊 Model: gpt-5-mini (optimized for cost)`);
  console.log(`⏱️  Timeout: 30s`);
});
