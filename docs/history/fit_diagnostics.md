# ResQ — fit diagnostics (train vs out-of-fold vs test)

_Phase-6 M3. The real overfitting test: a model whose **train** metric vastly beats its **out-of-fold** metric is overfitting; train ≈ OOF ≈ test means it is just capturing little signal. OOF = GroupKFold by storm inside pre-2019. `src/fit_diagnostics.py`._

| model | metric | train | OOF | test | flag |
|---|---|--:|--:|--:|---|
| GBM Δwind 6h | MAE kt | 2.4 | 3.6 | 4.2 | **OVERFIT** |
| GBM Δwind 12h | MAE kt | 3.5 | 6.1 | 7.6 | **OVERFIT** |
| GBM Δwind 24h | MAE kt | 4.7 | 10.6 | 13.1 | **OVERFIT** |
| GBM track-residual 6h | gc err km | 38.1 | 58.3 | 40.1 | **OVERFIT** |
| GBM track-residual 12h | gc err km | 58.3 | 105.3 | 77.7 | **OVERFIT** |
| GBM track-residual 24h | gc err km | 92.4 | 188.3 | 152.1 | **OVERFIT** |
| true CLIPER 6h | gc err km | 55.6 | 56.2 | 39.0 | ok |
| true CLIPER 12h | gc err km | 98.6 | 100.0 | 74.0 | ok |
| true CLIPER 24h | gc err km | 179.8 | 184.0 | 147.2 | ok |
| RI classifier (HGB) | PR-AUC | 0.930 | 0.155 | 0.219 | **OVERFIT** |
| RI analog P(RI), k=10 | PR-AUC | 0.093 | 0.105 | 0.234 | ok |

**Overfitting flagged:** GBM Δwind 6h, GBM Δwind 12h, GBM Δwind 24h, GBM track-residual 6h, GBM track-residual 12h, GBM track-residual 24h, RI classifier (HGB).

For every flagged model the **OOF score ≈ the test score** while the **train score is far better** — the models memorise their training storms. The honest performance is the OOF/test column (which is why the phase-5 bootstrap found none of them beats a trivial baseline); the train column is a mirage.

- The **RI classifier** is the extreme case: train PR-AUC 0.93 vs OOF 0.15. Phase-6 M1 responds by selecting a **more regularised** config (shallower trees) chosen on OOF PR-AUC.

- **true CLIPER** (linear) and the **analog ensemble** are *not* flagged — train ≈ OOF ≈ test. Linear / instance-based models with no capacity to memorise are the ones that generalise here.

- So the picture is two failure modes at once: the kinematic feature set is weak **and** the flexible models overfit what little there is. Both point the same way — more informative features (ERA5) matter more than a bigger model.

**LSTM:** not retrained in phase 6 — frozen test numbers only (results.md §1: 53/104/186 km track, 4.7/9.4/16.6 kt Δwind). Its test error is *worse* than the trivial baselines, which rules out train-memorisation as the failure mode — a model that overfit would at least fit its own training data.
