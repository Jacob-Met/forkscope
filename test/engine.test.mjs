import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseCsv, multiverse, runSpec, BASELINE, specId, parseSpecId, deviations, checkCitations, facts } from "../src/multiverse.mjs";
import { tPValue } from "../src/stats.mjs";
import { pair, notebook } from "../src/notebook.mjs";
import { extractJson, chat, propose, explain } from "../src/nemotron.mjs";
import { verifyNotebook } from "../server.mjs";

const csv = readFileSync(new URL("../data/mtcars.csv", import.meta.url), "utf8");
const rows = parseCsv(csv);

test("dataset is the 32 mtcars rows with known group means", () => {
  assert.equal(rows.length, 32);
  const m = (am) => rows.filter((r) => r.am === am).reduce((s, r) => s + r.mpg, 0) / rows.filter((r) => r.am === am).length;
  assert.ok(Math.abs(m(1) - 24.3923) < 1e-3 && Math.abs(m(0) - 17.1474) < 1e-3);
});
test("t distribution p-values match reference values", () => {
  assert.ok(Math.abs(tPValue(2.0, 10) - 0.073388) < 1e-5);
  assert.ok(Math.abs(tPValue(4.106, 30) - 0.000285) < 1e-5);
});
test("baseline reproduces the published lm(mpg ~ am) estimate", () => {
  const b = runSpec(rows, BASELINE);
  assert.ok(Math.abs(b.coef - 7.244939) < 1e-5 && Math.abs(b.p - 0.000285) < 1e-5);
  assert.equal(b.verdict, "supported");
});
test("adjusting for weight erases the effect (lm(mpg ~ am + wt))", () => {
  const r = runSpec(rows, { ...BASELINE, covariates: ["wt"] });
  assert.ok(Math.abs(r.coef + 0.02362) < 1e-4 && r.p > 0.9);
  assert.equal(r.verdict, "unsupported");
});
test("multiverse enumerates 2048 specs and finds one-choice reversals", () => {
  const mv = multiverse(rows);
  assert.equal(mv.total, 2048);
  assert.equal(mv.supported + mv.unsupported + mv.reversed, 2048);
  assert.equal(mv.minSize, 1);
  assert.ok(mv.minimal.every((r) => r.deviations === 1 && r.verdict !== "supported"));
  assert.ok(mv.minimal.some((r) => r.id.endsWith("c=wt")));
});
test("allow lists restrict the multiverse but keep the baseline", () => {
  const mv = multiverse(rows, { outliers: [], transforms: ["log"], ses: [], covariates: ["wt"] });
  assert.equal(mv.total, 4);
  assert.ok(mv.results.some((r) => r.id === specId(BASELINE)));
});
test("spec ids round trip and reject unknown choices", () => {
  const s = { outlier: "z2_wt", transform: "inv", se: "hc3", covariates: ["hp", "wt"] };
  assert.deepEqual(parseSpecId(specId(s)), s); assert.equal(deviations(s), 5);
  assert.throws(() => parseSpecId("o=evil|t=none|s=classic|c=-"));
});
test("paired notebooks differ only in the choices cell", () => {
  const p = pair(csv, specId(BASELINE), "o=none|t=none|s=classic|c=wt");
  assert.ok(p.diff.length >= 1 && p.diff.every((d) => d.cell === 1));
  assert.ok(p.diff.some((d) => /"wt"/.test(d.b)));
});
test("python notebook executes and matches the JS engine", async () => {
  const spec = { outlier: "tukey_mpg", transform: "inv", se: "hc3", covariates: ["wt", "qsec"] };
  const v = await verifyNotebook(notebook(spec, csv, "t"), runSpec(rows, spec));
  assert.equal(v.ran, true, v.error); assert.equal(v.match, true);
});
test("citation checker accepts true numbers and rejects invented ones", () => {
  const f = facts(rows, { ...BASELINE, covariates: ["wt"] });
  const c = checkCitations([{ fact: "corr_am_wt", value: f.corr_am_wt }, { fact: "corr_am_wt", value: 0.5 }, { fact: "made_up", value: 1 }], f);
  assert.deepEqual(c.map((x) => x.ok), [true, false, false]);
});

// Nemotron adapter, no network: fetch is injected.
const fakeFetch = (content, ok = true) => { const f = async (url, init) => { f.last = { url, init }; return { ok, status: ok ? 200 : 500, json: async () => ({ choices: [{ message: { content, reasoning_content: "ignored" } }] }) }; }; return f; };
test("adapter degrades gracefully without a key", async () => {
  delete process.env.NEBIUS_API_KEY;
  await assert.rejects(chat([{ role: "user", content: "x" }]), (e) => e.code === "NO_KEY" && /NEBIUS_API_KEY/.test(e.message));
});
test("adapter calls Token Factory with Bearer auth, a Nemotron model and enough tokens", async () => {
  process.env.NEBIUS_API_KEY = "test-key";
  const f = fakeFetch("hello");
  assert.equal(await chat([{ role: "user", content: "hi" }], { fetchImpl: f }), "hello");
  assert.match(f.last.url, /tokenfactory\.nebius\.com\/v1\/chat\/completions$/);
  const body = JSON.parse(f.last.init.body);
  assert.match(body.model, /nemotron/i); assert.ok(body.max_tokens >= 1500);
  assert.equal(f.last.init.headers.authorization, "Bearer test-key");
});
test("extractJson tolerates code fences", () => assert.deepEqual(extractJson('```json\n{"a":1}\n```'), { a: 1 }));
test("propose drops ids the engine does not know and keeps only defensible ones", async () => {
  process.env.NEBIUS_API_KEY = "test-key";
  const reply = JSON.stringify({ covariates: { wt: { defensible: true, reason: "r" }, hp: { defensible: false, reason: "r" }, bogus: { defensible: true, reason: "r" } }, outliers: {}, transforms: {}, ses: {} });
  const r = await propose({ fetchImpl: fakeFetch(reply) });
  assert.deepEqual(r.allow.covariates, ["wt"]); assert.deepEqual(r.rejectedIds, ["covariates.bogus"]);
});
test("explain marks fabricated citations as wrong", async () => {
  process.env.NEBIUS_API_KEY = "test-key";
  const spec = { ...BASELINE, covariates: ["wt"] }, f = facts(rows, spec);
  const reply = JSON.stringify({ explanation: "e", caveat: "c", cited: [{ fact: "corr_am_wt", value: f.corr_am_wt }, { fact: "fork_p", value: 0.001 }] });
  const r = await explain({ baselineId: specId(BASELINE), forkId: specId(spec), facts: f, baseline: runSpec(rows, BASELINE), fork: runSpec(rows, spec) }, { fetchImpl: fakeFetch(reply) });
  assert.deepEqual(r.checks.map((c) => c.ok), [true, false]);
});
