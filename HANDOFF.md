# Drug Shortage Radar — Session Handoff

Pick-up notes. Read this + `PLAN.md` + memory `project_shortage_radar_modeling.md`.

## Where the model stands (val 2023–2024, test LOCKED)

LightGBM, retrained after a token-set label fix (+10% positives) and overfitting hardening.
- **PR-AUC ≈ 0.113, ROC-AUC ≈ 0.81** (train PR-AUC 0.19 — gap narrowed by regularization).
- **Watchlist precision@K (deployment metric):** top-20 = 45%, **top-50 = 46% (20× lift)**,
  top-100 = 31%, top-200 = 34%, top-500 = 26%.
  - *Honest caveat:* precision@K counts drug-MONTHS; the top-50 is ~10 distinct chronic
    drugs (rifampin IV, lidocaine HCl IV, potassium chloride, epinephrine) across months.
- **Lead time:** monthly top-50 flags 84/545 onsets in advance (15% catch rate),
  mean ~4.3 months warning (often the full 6). Catch-rate is the real weak spot.
- **Calibration:** isotonic (in-sample val), Brier 0.151→0.021, mean pred ≈ base rate.
- **Overfitting guards:** num_leaves 16 / min_child_samples 100; final model trains a FIXED
  CV-derived tree count (79) — does NOT early-stop on val, so val is clean held-out.
- **Honest test expectation:** PR-AUC ~0.08–0.11, precision@50 ~30–45% (CV PR-AUC ≈ 0.07).
- Panel: train 235,763 / 5,189 pos · val 108,627 / 2,475 · test 54,592 / 1,181.
- Full report: `reports/validation.md` (incl. §7 comparison to prior work).

## Pipeline (reconciled, end-to-end — run in this order)
1. `python scripts/build_panel.py [--force]` → `data/processed/features_with_labels.parquet`
2. `python scripts/train.py [--skip-cv]` → trains LightGBM+logistic, saves
   `models/{lgbm_final.txt,feature_cols.json,calibrator.pkl}` + `predictions_val.parquet`
3. `python scripts/evaluate.py` → writes `reports/validation.md`
4. `python scripts/build_figures.py` → `reports/figures/*.png` + `figures.html`
   (also writes `predictions_test.parquet` = frozen test, display-only)
5. `python app.py` → 7-tab Dash dashboard
6. `python -m pytest tests/ -q` → 42/42 green

## Presentation layer (this session)
- `src/figures.py` — 18 pure Plotly builders (EDA, val-vs-test performance, benchmark),
  shared light "report" template. Reused by both app and export script.
- `app.py` — light FLATLY theme, tabs: **Overview · Data & EDA · Watchlist · Drug Detail ·
  Model Performance · Benchmark · Model Card** (renders both validation.md + test_LOCKED.md).
- `scripts/build_figures.py` — PNG (kaleido) + HTML export for the README.
- **Dependency note:** PNG export needs Chrome for kaleido v1 — run `plotly_get_chrome` once
  (installed to `~/.local/share/choreographer/`). The live app needs no Chrome.

Test stays LOCKED: `scripts/eval_test_LOCKED.py`, only on explicit request, only once
model+features are final (they are NOT — label matching just changed).

## What was done this session
1. **Reconciled a mid-refactor repo** (two naming conventions). Canonical artifacts under
   `data/processed/`; fixed `train.py`/`evaluate.py`/`explain.py`/`app.py` paths + columns.
2. **Phase 6 evaluation + lead-time** built (`src/evaluate.py`): per-onset earliest-catch
   in a monthly top-K watchlist → mean ~4.1 mo advance warning.
3. **Isotonic calibration** so dashboard scores are real probabilities.
4. **Investigated supplier-exit features → dead end** (NDC is a current snapshot; no
   historical exits). Kept the computation, excluded from FEATURE_COLS with a comment.
5. **Token-set label matching** for combination drugs (`src/panel.py::_tokset`) —
   recovered Adderall + ~10% more positives; precision@50 30%→58%.
6. **SHAP drivers wired into the app** Drug Detail tab (`src/explain.py`).
7. Tests aligned to real column names; adderall test rewritten (old window was wrong).

## Open / next
- **Catch-rate (15%)** is the honest weak point — widen K, or report recall@6mo, or a
  per-DRUG (not drug-month) watchlist so precision@K isn't inflated by repeats.
- Label tail: 24% of shortage names still unmatched (brand-only / salts absent from NDC).
- `reason_for_recall` text-mining + chronic-shortage profile features (Tier-2, optional).
- When final: run the locked test ONCE.
