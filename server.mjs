import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseCsv, multiverse, runSpec, parseSpecId, specId, BASELINE, CONCLUSION, COVARIATES, OUTLIERS, TRANSFORMS, SES, facts } from "./src/multiverse.mjs";
import { pair, pythonSource } from "./src/notebook.mjs";
import * as nem from "./src/nemotron.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const csvText = readFileSync(join(here, "data/mtcars.csv"), "utf8");
const rows = parseCsv(csvText);
const send = (res, code, body, type = "application/json") => { res.writeHead(code, { "content-type": type, "cache-control": "no-store" }); res.end((typeof body === "string" || Buffer.isBuffer(body)) ? body : JSON.stringify(body)); };
const readBody = (req) => new Promise((ok) => { let d = ""; req.on("data", (c) => (d += c)); req.on("end", () => { try { ok(d ? JSON.parse(d) : {}); } catch { ok({}); } }); });
const slim = (r) => ({ id: r.id, n: r.n, coef: r.coef, se: r.se, t: r.t, p: r.p, favors: r.favors, favorT: r.favorT, verdict: r.verdict, deviations: r.deviations, dropped: r.dropped });

function runPython(src) {
  return new Promise((ok) => execFile("python3", ["-I", "-c", src], { timeout: 20000, maxBuffer: 1 << 20 }, (err, out, errOut) => ok({ err: err && (errOut || err.message), out })));
}
export async function verifyNotebook(nb, expected) {
  const { err, out } = await runPython(pythonSource(nb));
  if (err) return { ran: false, error: String(err).slice(0, 300) };
  const j = JSON.parse(out.slice(0, out.indexOf("\nVERDICT")));
  const close = (a, b) => Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(b));
  return { ran: true, python: { coef: j.coef, p: j.p, n: j.n, verdict: j.verdict }, engine: { coef: expected.coef, p: expected.p, n: expected.n, verdict: expected.verdict }, match: close(j.coef, expected.coef) && close(j.p, expected.p) && j.n === expected.n && j.verdict === expected.verdict };
}

let cache = null;
const allowKey = (a) => JSON.stringify(a || {});
function getMv(allow) { const k = allowKey(allow); if (!cache || cache.k !== k) cache = { k, mv: multiverse(rows, allow) }; return cache.mv; }

export const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://x"); const p = url.pathname;
  try {
    if (req.method === "GET" && (p === "/" || p === "/index.html")) return send(res, 200, readFileSync(join(here, "public/index.html")), "text/html; charset=utf-8");
    if (p === "/api/status") return send(res, 200, { nemotron: nem.hasKey(), model: nem.MODEL, conclusion: CONCLUSION, rows: rows.length, vocab: { outliers: OUTLIERS, transforms: TRANSFORMS, ses: SES, covariates: COVARIATES } });
    if (p === "/api/data") return send(res, 200, csvText, "text/csv");
    if (p === "/api/multiverse") {
      const body = req.method === "POST" ? await readBody(req) : {};
      const mv = getMv(body.allow);
      return send(res, 200, { alpha: mv.alpha, baseline: slim(mv.baseline), total: mv.total, supported: mv.supported, unsupported: mv.unsupported, reversed: mv.reversed, minSize: mv.minSize, minimal: mv.minimal.map(slim), singles: mv.singles.map(slim), specs: mv.results.map(slim) });
    }
    if (p === "/api/specs.csv") {
      const mv = getMv(null); const h = "id,n,coef,se,t,p,favors,verdict,deviations,dropped\n";
      return send(res, 200, h + mv.results.map((r) => [r.id, r.n, r.coef, r.se, r.t, r.p, r.favors, r.verdict, r.deviations, `"${r.dropped.join("; ")}"`].join(",")).join("\n") + "\n", "text/csv");
    }
    if (p === "/api/pair") {
      const a = url.searchParams.get("a") || specId(BASELINE), b = url.searchParams.get("b");
      const pr = pair(csvText, a, b); const ra = runSpec(rows, parseSpecId(a)), rb = runSpec(rows, parseSpecId(b));
      if (url.searchParams.get("download")) return send(res, 200, pr[url.searchParams.get("download") === "a" ? "a" : "b"], "application/x-ipynb+json");
      return send(res, 200, { diff: pr.diff, a: slim(ra), b: slim(rb), source: { a: pythonSource(pr.a).length, b: pythonSource(pr.b).length } });
    }
    if (p === "/api/verify" && req.method === "POST") {
      const { a, b } = await readBody(req); const pr = pair(csvText, a || specId(BASELINE), b);
      const [va, vb] = await Promise.all([verifyNotebook(pr.a, runSpec(rows, parseSpecId(a || specId(BASELINE)))), verifyNotebook(pr.b, runSpec(rows, parseSpecId(b)))]);
      return send(res, 200, { a: va, b: vb });
    }
    if (p === "/api/propose" && req.method === "POST") {
      try { const r = await nem.propose(); cache = null; return send(res, 200, r); }
      catch (e) { return send(res, e.code === "NO_KEY" ? 503 : 502, { error: e.message }); }
    }
    if (p === "/api/explain" && req.method === "POST") {
      const { a, b } = await readBody(req); const sa = parseSpecId(a || specId(BASELINE)), sb = parseSpecId(b);
      try {
        const r = await nem.explain({ baselineId: specId(sa), forkId: specId(sb), facts: facts(rows, sb), baseline: runSpec(rows, sa), fork: runSpec(rows, sb) });
        return send(res, 200, r);
      } catch (e) { return send(res, e.code === "NO_KEY" ? 503 : 502, { error: e.message }); }
    }
    send(res, 404, { error: "not found" });
  } catch (e) { send(res, 400, { error: e.message }); }
});
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4413);
  server.listen(port, () => console.log(`forkscope on http://localhost:${port}  (Nemotron ${nem.hasKey() ? "enabled: " + nem.MODEL : "disabled: set NEBIUS_API_KEY"})`));
}
