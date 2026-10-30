import { ols, mean, sd, quantile, corr } from "./stats.mjs";

export const CONCLUSION = "Manual transmission cars get better fuel economy than automatics.";
export const COVARIATES = { wt: "weight (1000 lb)", hp: "gross horsepower", cyl: "cylinders (as a number)", disp: "displacement (cu in)", qsec: "quarter-mile time (s)", drat: "rear axle ratio" };
export const OUTLIERS = {
  none: "keep every car",
  tukey_mpg: "drop mpg outside Tukey fences (1.5 IQR)",
  z2_mpg: "drop |z| > 2 on mpg",
  z2_wt: "drop |z| > 2 on weight",
};
export const TRANSFORMS = { none: "mpg", log: "log(mpg)", sqrt: "sqrt(mpg)", inv: "gallons per 100 miles (100/mpg)" };
export const SES = { classic: "classical standard errors", hc3: "HC3 robust standard errors" };
export const BASELINE = { outlier: "none", transform: "none", se: "classic", covariates: [] };
export const ALPHA = 0.05;

export function parseCsv(text) {
  const [head, ...lines] = text.trim().split(/\r?\n/); const cols = head.split(",");
  return lines.map((l) => { const c = l.split(","); const r = {}; cols.forEach((k, i) => (r[k] = i === 0 ? c[i] : Number(c[i]))); return r; });
}
export const specId = (s) => `o=${s.outlier}|t=${s.transform}|s=${s.se}|c=${[...s.covariates].sort().join("+") || "-"}`;
export function parseSpecId(id) {
  const m = /^o=(\w+)\|t=(\w+)\|s=(\w+)\|c=(.*)$/.exec(id); if (!m) throw new Error("bad spec id");
  const s = { outlier: m[1], transform: m[2], se: m[3], covariates: m[4] === "-" ? [] : m[4].split("+") };
  if (!OUTLIERS[s.outlier] || !TRANSFORMS[s.transform] || !SES[s.se] || s.covariates.some((c) => !COVARIATES[c])) throw new Error("unknown choice in spec id");
  return s;
}
export const deviations = (s) => (s.outlier !== "none") + (s.transform !== "none") + (s.se !== "classic") + s.covariates.length;
const TF = { none: (v) => v, log: Math.log, sqrt: Math.sqrt, inv: (v) => 100 / v };

export function keepRows(rows, outlier) {
  if (outlier === "none") return rows;
  if (outlier === "tukey_mpg") { const v = rows.map((r) => r.mpg), q1 = quantile(v, 0.25), q3 = quantile(v, 0.75), i = q3 - q1; return rows.filter((r) => r.mpg >= q1 - 1.5 * i && r.mpg <= q3 + 1.5 * i); }
  const col = outlier === "z2_mpg" ? "mpg" : "wt"; const v = rows.map((r) => r[col]), m = mean(v), s = sd(v);
  return rows.filter((r) => Math.abs((r[col] - m) / s) <= 2);
}
export function runSpec(rows, spec, alpha = ALPHA) {
  const kept = keepRows(rows, spec.outlier);
  const X = kept.map((r) => [1, r.am, ...spec.covariates.map((c) => r[c])]);
  const y = kept.map((r) => TF[spec.transform](r.mpg));
  const f = ols(X, y, 1, spec.se);
  const favors = (spec.transform === "inv" ? f.coef < 0 : f.coef > 0) ? "manual" : "automatic";
  const favorT = favors === "manual" ? Math.abs(f.t) : -Math.abs(f.t);
  const verdict = f.p < alpha ? (favors === "manual" ? "supported" : "reversed") : "unsupported";
  return { id: specId(spec), spec, n: kept.length, dropped: rows.filter((r) => !kept.includes(r)).map((r) => r.car), ...f, favors, favorT, verdict, deviations: deviations(spec) };
}
const subsets = (a) => a.reduce((acc, v) => acc.concat(acc.map((s) => [...s, v])), [[]]);
export function enumerate(allow = {}) {
  const pick = (all, want, base) => { const keys = Object.keys(all); const w = want ? keys.filter((k) => want.includes(k) || k === base) : keys; return w; };
  const out = pick(OUTLIERS, allow.outliers, "none"), tr = pick(TRANSFORMS, allow.transforms, "none"), se = pick(SES, allow.ses, "classic");
  const cov = Object.keys(COVARIATES).filter((c) => !allow.covariates || allow.covariates.includes(c));
  const specs = [];
  for (const o of out) for (const t of tr) for (const s of se) for (const c of subsets(cov)) specs.push({ outlier: o, transform: t, se: s, covariates: c });
  return specs;
}
export function multiverse(rows, allow = {}, alpha = ALPHA) {
  const results = enumerate(allow).map((s) => runSpec(rows, s, alpha));
  const baseline = runSpec(rows, BASELINE, alpha);
  const count = (v) => results.filter((r) => r.verdict === v).length;
  const flips = results.filter((r) => r.verdict !== "supported");
  const minSize = flips.length ? Math.min(...flips.map((r) => r.deviations)) : null;
  const minimal = flips.filter((r) => r.deviations === minSize).sort((a, b) => b.p - a.p);
  const singles = results.filter((r) => r.deviations === 1).sort((a, b) => a.id.localeCompare(b.id));
  return { alpha, baseline, total: results.length, supported: count("supported"), unsupported: count("unsupported"), reversed: count("reversed"), minSize, minimal, singles, results };
}
// Facts the engine computes so a model's explanation can be checked against them.
export function facts(rows, spec) {
  const f = {}; const m = rows.map((r) => r.am);
  for (const c of spec.covariates) {
    f[`corr_am_${c}`] = corr(m, rows.map((r) => r[c]));
    f[`mean_${c}_manual`] = mean(rows.filter((r) => r.am === 1).map((r) => r[c]));
    f[`mean_${c}_automatic`] = mean(rows.filter((r) => r.am === 0).map((r) => r[c]));
  }
  f.mean_mpg_manual = mean(rows.filter((r) => r.am === 1).map((r) => r.mpg));
  f.mean_mpg_automatic = mean(rows.filter((r) => r.am === 0).map((r) => r.mpg));
  const b = runSpec(rows, BASELINE), x = runSpec(rows, spec);
  f.baseline_coef = b.coef; f.baseline_p = b.p; f.fork_coef = x.coef; f.fork_p = x.p; f.fork_n = x.n; f.baseline_n = b.n;
  return f;
}
export function checkCitations(cited, truth) {
  return (cited || []).map((c) => {
    const real = truth[c?.fact]; const val = Number(c?.value);
    const ok = real !== undefined && Number.isFinite(val) && Math.abs(val - real) <= Math.max(0.02, Math.abs(real) * 0.02);
    return { fact: c?.fact, claimed: c?.value, actual: real ?? null, ok };
  });
}
