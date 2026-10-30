# Forkscope

Find the smallest defensible analysis choices that reverse a statistical conclusion, then compare the two analyses as paired, executable notebooks.

Forkscope runs a **multiverse analysis** on a small real dataset (the 32-car Motor Trend `mtcars` road tests). The claim under test: *manual transmission cars get better fuel economy*, estimated as the coefficient of `am` on `mpg`. The engine runs every combination of outlier rule, outcome transform, standard-error type and covariate set (2,048 specifications), draws the specification curve, and searches for the **minimal set of choices that flips the verdict**. In the default data, 403 of 2,048 specifications support the claim and a single choice (adjusting for weight) takes the effect from +7.24 mpg to about 0.

For any flip you get two paired Jupyter notebooks (`.ipynb`) that differ only in the lines that implement the choice, a line diff, and a button that runs both in fresh Python processes (standard library only) and checks they match the JavaScript engine.

## Quick start

Requires Node 22+ and `python3` (for the notebook check). No npm install; there are no runtime dependencies.

```sh
export NEBIUS_API_KEY=your_token_factory_key   # optional: enables the Nemotron features
node server.mjs                                # PORT=4413 by default
# open http://localhost:4413
node --test test/                              # 15 tests, no network needed
```

Without a key everything deterministic still works; the two Nemotron buttons show a clear "key not set" message.

## How it uses NVIDIA Nemotron on Nebius Token Factory

The server calls Token Factory's OpenAI-compatible endpoint (`https://api.tokenfactory.nebius.com/v1/chat/completions`) with `nvidia/nemotron-3-super-120b-a12b` (override with `NEBIUS_MODEL`). The key is read from `NEBIUS_API_KEY` on the server; the browser never sees it. Code: `src/nemotron.mjs`.

Why this model: the tasks need statistical judgment and structured JSON output, not a tiny classifier. The Super model is a 120B mixture-of-experts with 12B active parameters, which keeps calls to a few seconds to ~20 seconds while following the JSON contract reliably.

Two features, both "model proposes, engine verifies":

1. **Vet the forks** (`POST /api/propose`). Nemotron is shown the claim and every available choice (outlier rules, transforms, SE types, covariates) and says which a reasonable analyst could defend, with a one-sentence reason each. The engine ignores ids it does not know, uses only the choices the model marked defensible (the baseline is always kept), and re-runs the whole multiverse on that set.
2. **Explain a flip** (`POST /api/explain`). Given a baseline and a forked analysis, Nemotron writes a short explanation and must cite numbers from a facts table computed by the engine (group means, correlation of `am` with `wt`, coefficients). Every cited number is compared to the engine's value and marked correct or wrong in the UI. Invented numbers are flagged, not hidden.

All estimates, p-values, the minimal-flip search and the notebooks come from deterministic code, never from the model.

## Layout

- `src/stats.mjs` OLS, HC3, t/p values. `src/multiverse.mjs` specifications, search, fact table, citation checker.
- `src/notebook.mjs` builds paired notebooks and the line diff. `src/nemotron.mjs` Token Factory adapter.
- `server.mjs` HTTP API and static UI (`public/index.html`). `test/engine.test.mjs` tests (engine, Python notebook match, adapter with a stubbed fetch).

## Limits

- One dataset and one claim; this is a demonstration of the method, not a general statistics package. n=32.
- The multiverse is a fixed menu of choices. "Defensible" is Nemotron's judgment, so it can be wrong or inconsistent between runs; the final vet is always shown with reasons.
- A flip shows fragility, not which analysis is correct. Weight genuinely confounds transmission here, which is why that fork is the most defensible one.
- Citation checking verifies the numbers the model cites, not the quality of its reasoning.
- The notebook check needs `python3` on the PATH.

## License

MIT, see `LICENSE`.
