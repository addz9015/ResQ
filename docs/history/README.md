# Correction history

ResQ was built in phases, and each phase fixed something the previous one got
wrong. These are the working notes for that process — kept because *how* the
project was corrected is part of the evidence, not clutter. The load-bearing
findings live one level up in [`reports/`](../../reports/); this folder is the
audit trail behind them.

Read [`reports/audit.md`](../../reports/audit.md),
[`reports/results.md`](../../reports/results.md),
[`reports/significance.md`](../../reports/significance.md) and
[`reports/limitations.md`](../../reports/limitations.md) first — they are the
current state. Come here for the "why".

| file | phase | what it covers |
|---|---|---|
| [`corrections.md`](corrections.md) | 3 | the four phase-3 corrections: unfair model-comparison n, calibration that touched the test set, track models predicting position instead of the CLIPER residual, interpolated targets |
| [`phase4.md`](phase4.md) | 4 | true CLIPER (OLS) vs persistence-of-motion, cross-conformal calibration, the combined P(RI) signal, the ERA5 fetch layer, the first dashboard |
| [`phase6.md`](phase6.md) | 6 | dashboard tabs; M1 RI-ensemble tuning (pre-2019 OOF only), M2 learning curve, M3 overfitting check |
| [`fit_diagnostics.md`](fit_diagnostics.md) | 6 (M3) | train vs out-of-fold vs test for every fitted model — which ones overfit |
| [`phase7.md`](phase7.md) | 7 | end-to-end wiring, one-command startup; the archived-IMD-bulletin ingestion attempt (0 obtained — reason recorded) |
| [`phase8.md`](phase8.md) | 8 | the four citizen tabs — Preparation Checklist, District Cyclone History, Official Warnings (paste-and-explain), retrieval chatbot |
| [`phase9.md`](phase9.md) | 9 | NDMA guidance harvested via the Drupal REST API (51 verbatim items); checklist rules map + bulletin prefill; Groq chatbot with a post-generation guard; real-bulletin parser test |
| [`phase10.md`](phase10.md) | 10 | verbatim mode for preparedness answers (assembled from exact NDMA strings; Groq only picks item ids); negation / meaning-inversion guard; `reports/guard_discards.jsonl` |

Some cross-references inside these files point at each other by their old
top-level `reports/…` paths — they are historical documents and were left as
written.
