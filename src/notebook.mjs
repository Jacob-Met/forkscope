import { COVARIATES, OUTLIERS, TRANSFORMS, SES, CONCLUSION, parseSpecId } from "./multiverse.mjs";

// Pure-stdlib Python mirror of src/stats.mjs + src/multiverse.mjs#runSpec.
export const PY_ENGINE = `import csv, io, math, json, statistics

def betacf(a, b, x):
    FP = 1e-300; qab = a + b; qap = a + 1; qam = a - 1
    c = 1.0; d = 1 - qab * x / qap
    if abs(d) < FP: d = FP
    d = 1 / d; h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = d if abs(d) >= FP else FP
        c = 1 + aa / c; c = c if abs(c) >= FP else FP
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = d if abs(d) >= FP else FP
        c = 1 + aa / c; c = c if abs(c) >= FP else FP
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 3e-14: break
    return h

def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2): return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b

def invert(M):
    n = len(M); A = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-12: raise ValueError("singular design matrix")
        A[c], A[p] = A[p], A[c]; pv = A[c][c]; A[c] = [v / pv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]; A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [r[n:] for r in A]

def ols(X, y, k, se):
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    inv = invert(XtX)
    beta = [sum(inv[a][j] * Xty[j] for j in range(p)) for a in range(p)]
    e = [y[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    df = n - p
    if se == "hc3":
        meat = [[0.0] * p for _ in range(p)]
        for i in range(n):
            h = sum(X[i][a] * inv[a][b] * X[i][b] for a in range(p) for b in range(p))
            w = (e[i] / (1 - h)) ** 2
            for a in range(p):
                for b in range(p): meat[a][b] += w * X[i][a] * X[i][b]
        tmp = [[sum(inv[r][m] * meat[m][j] for m in range(p)) for j in range(p)] for r in range(p)]
        var = sum(tmp[k][m] * inv[m][k] for m in range(p))
    else:
        var = sum(v * v for v in e) / df * inv[k][k]
    s = math.sqrt(var); t = beta[k] / s
    return dict(coef=beta[k], se=s, t=t, df=df, p=betai(df / 2, 0.5, df / (df + t * t)))

def quantile(a, q):
    s = sorted(a); h = (len(s) - 1) * q; lo = math.floor(h)
    return s[lo] + (h - lo) * ((s[lo + 1] if lo + 1 < len(s) else s[lo]) - s[lo])

def keep_rows(rows, rule):
    if rule == "none": return rows
    if rule == "tukey_mpg":
        v = [r["mpg"] for r in rows]; q1, q3 = quantile(v, .25), quantile(v, .75); i = q3 - q1
        return [r for r in rows if q1 - 1.5 * i <= r["mpg"] <= q3 + 1.5 * i]
    col = "mpg" if rule == "z2_mpg" else "wt"
    v = [r[col] for r in rows]; m, s = statistics.fmean(v), statistics.stdev(v)
    return [r for r in rows if abs((r[col] - m) / s) <= 2]

TF = {"none": lambda v: v, "log": math.log, "sqrt": math.sqrt, "inv": lambda v: 100 / v}

def run(rows, spec, alpha=0.05):
    kept = keep_rows(rows, spec["outlier"])
    X = [[1.0, r["am"]] + [r[c] for c in spec["covariates"]] for r in kept]
    y = [TF[spec["transform"]](r["mpg"]) for r in kept]
    f = ols(X, y, 1, spec["se"])
    manual = f["coef"] < 0 if spec["transform"] == "inv" else f["coef"] > 0
    favors = "manual" if manual else "automatic"
    verdict = ("supported" if manual else "reversed") if f["p"] < alpha else "unsupported"
    return dict(n=len(kept), dropped=[r["car"] for r in rows if r not in kept], favors=favors, verdict=verdict, **f)

def load(text):
    return [{k: (v if k == "car" else float(v)) for k, v in r.items()} for r in csv.DictReader(io.StringIO(text))]
`;

const q = (v) => JSON.stringify(v);
export function specCell(spec, label) {
  return [
    `# ${label}`,
    "SPEC = {",
    `    "outlier": ${q(spec.outlier)},      # ${OUTLIERS[spec.outlier]}`,
    `    "transform": ${q(spec.transform)},      # outcome: ${TRANSFORMS[spec.transform]}`,
    `    "se": ${q(spec.se)},      # ${SES[spec.se]}`,
    `    "covariates": ${q(spec.covariates)},      # adjusted for: ${spec.covariates.map((c) => COVARIATES[c]).join(", ") || "nothing"}`,
    "}",
  ].join("\n");
}
const lines = (s) => s.split("\n").map((l, i, a) => (i < a.length - 1 ? l + "\n" : l));
const code = (s) => ({ cell_type: "code", metadata: {}, execution_count: null, outputs: [], source: lines(s) });
const md = (s) => ({ cell_type: "markdown", metadata: {}, source: lines(s) });

export function notebook(spec, csvText, label) {
  return {
    nbformat: 4, nbformat_minor: 5,
    metadata: { kernelspec: { name: "python3", display_name: "Python 3", language: "python" }, language_info: { name: "python" }, forkscope: { label } },
    cells: [
      md(`# ${label}\n\nConclusion under test: **${CONCLUSION}**\n\nThis notebook needs only the Python standard library. Run all cells; the last cell prints the estimate of the effect of manual transmission (\`am\`).`),
      code(specCell(spec, "THE ANALYSIS CHOICES (the only cell that differs between paired notebooks)")),
      code("CSV = " + q(csvText)),
      code(PY_ENGINE),
      code(`rows = load(CSV)\nresult = run(rows, SPEC)\nprint(json.dumps(result, indent=2))\nprint("VERDICT:", result["verdict"], "(p =", round(result["p"], 4), ")")`),
    ],
  };
}
export function pythonSource(nb) { return nb.cells.filter((c) => c.cell_type === "code").map((c) => c.source.join("")).join("\n\n"); }

export function pair(csvText, idA, idB) {
  const a = parseSpecId(idA), b = parseSpecId(idB);
  const A = notebook(a, csvText, "Notebook A"), B = notebook(b, csvText, "Notebook B");
  const diff = [];
  A.cells.forEach((c, i) => {
    if (i === 0) return; // title cell carries only the A/B label
    const x = c.source.join("").split("\n"), y = B.cells[i].source.join("").split("\n");
    for (let j = 0; j < Math.max(x.length, y.length); j++) if (x[j] !== y[j]) diff.push({ cell: i, line: j + 1, a: x[j] ?? "", b: y[j] ?? "" });
  });
  return { a: A, b: B, diff };
}
