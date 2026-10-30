// Nemotron on Nebius Token Factory (OpenAI-compatible). The model proposes and explains; src/multiverse.mjs verifies.
import { OUTLIERS, TRANSFORMS, SES, COVARIATES, CONCLUSION, checkCitations } from "./multiverse.mjs";

export const BASE_URL = process.env.NEBIUS_BASE_URL || "https://api.tokenfactory.nebius.com/v1";
export const MODEL = process.env.NEBIUS_MODEL || "nvidia/nemotron-3-super-120b-a12b";
export const hasKey = () => Boolean(process.env.NEBIUS_API_KEY);

export async function chat(messages, { fetchImpl = fetch, maxTokens = 4000 } = {}) {
  if (!hasKey()) { const e = new Error("NEBIUS_API_KEY is not set on the server; Nemotron features are off. The deterministic engine still works."); e.code = "NO_KEY"; throw e; }
  const r = await fetchImpl(BASE_URL + "/chat/completions", {
    method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${process.env.NEBIUS_API_KEY}` },
    body: JSON.stringify({ model: MODEL, messages, max_tokens: maxTokens, temperature: 0.2 }),
  });
  if (!r.ok) throw new Error(`Token Factory returned HTTP ${r.status}`);
  const j = await r.json();
  return String(j.choices?.[0]?.message?.content ?? "");
}
export function extractJson(text) {
  const s = text.replace(/```(?:json)?/g, ""); const a = s.indexOf("{"), b = s.lastIndexOf("}");
  if (a < 0 || b <= a) throw new Error("model did not return JSON");
  return JSON.parse(s.slice(a, b + 1));
}
const vocab = () => ({ outliers: OUTLIERS, transforms: TRANSFORMS, ses: SES, covariates: COVARIATES });

export async function propose(opts) {
  const sys = "You are a careful applied statistician reviewing a multiverse analysis. Reply with one JSON object only.";
  const user = `Dataset: Motor Trend 1974 road tests of 32 cars (mtcars). Conclusion under test: "${CONCLUSION}" estimated as the coefficient of manual transmission (am=1) on mpg.
For each choice id below, say whether a reasonable analyst could defend using it (defensible true/false) and give a one-sentence reason. Be strict: only mark false if the choice is statistically unjustified for this question.
Choices: ${JSON.stringify(vocab())}
Return: {"outliers":{"<id>":{"defensible":bool,"reason":str}}, "transforms":{...}, "ses":{...}, "covariates":{...}}`;
  const raw = await chat([{ role: "system", content: sys }, { role: "user", content: user }], opts);
  const j = extractJson(raw); const v = vocab(); const accepted = {}, rejectedIds = [], reasons = {};
  for (const dim of Object.keys(v)) {
    accepted[dim] = [];
    for (const [id, o] of Object.entries(j[dim] || {})) {
      if (!(id in v[dim])) { rejectedIds.push(`${dim}.${id}`); continue; }
      reasons[`${dim}.${id}`] = { defensible: !!o?.defensible, reason: String(o?.reason ?? "") };
      if (o?.defensible) accepted[dim].push(id);
    }
    // Ids the model never mentioned stay out of the allow list; the baseline choice is always kept by the engine.
  }
  return { model: MODEL, allow: accepted, reasons, rejectedIds };
}

export async function explain(ctx, opts) {
  const { baselineId, forkId, facts: f, baseline, fork } = ctx;
  const sys = "You explain why one analysis choice changes a statistical result. Use ONLY the numbers in FACTS. Reply with one JSON object only.";
  const user = `Conclusion: "${CONCLUSION}"
Analysis A (${baselineId}): coef ${baseline.coef.toFixed(3)}, p ${baseline.p.toFixed(4)}, ${baseline.verdict}.
Analysis B (${forkId}): coef ${fork.coef.toFixed(3)}, p ${fork.p.toFixed(4)}, ${fork.verdict}, n=${fork.n}.
FACTS (computed by the app): ${JSON.stringify(f)}
Write 3 to 4 plain sentences explaining why B differs from A (confounding, lost data, variance, scale). Then list the FACTS you relied on.
Return: {"explanation": str, "cited": [{"fact": "<key from FACTS>", "value": <number>}], "caveat": str}`;
  const j = extractJson(await chat([{ role: "system", content: sys }, { role: "user", content: user }], opts));
  return { model: MODEL, explanation: String(j.explanation ?? ""), caveat: String(j.caveat ?? ""), checks: checkCitations(j.cited, f) };
}
