# Drug Shortage Early-Warning System — Execution Plan

**Goal:** Predict which US drug products will enter FDA-reported shortage within the next 6 months, before the shortage is posted. Deliverables: a reproducible data pipeline, a trained + backtested model, and a Plotly Dash dashboard showing a ranked risk watchlist with per-drug explanations.

**Capstone thesis:** Public FDA signals (manufacturing inspections, citations, compliance actions, recalls, market concentration, price pressure) contain early warning of supply disruption. Nobody assembles them into a per-drug forecast. We do.

**Working directory:** `/home/jin/Documents/GITHUB/drug-shortage-radar/` (this file lives there).

---

## Hard constraints (do not violate)

1. **No paid data, no scraping behind logins.** Every source below is free/public. Prefer endpoints that need no API key; openFDA works keyless at 240 req/min/IP (a free key raises limits if ever needed). The FDA Data Dashboard API needs a free authorization key — prefer its bulk Excel/CSV exports first; only fall back to requesting a key if exports are insufficient.
2. **Locked test set.** Temporal split (Phase 3). The test window is evaluated ONCE, at the very end, by the user's explicit request — never auto-run it during iteration. All model iteration happens on the validation window only.
3. **No feature leakage.** Every feature value at panel month `t` must be computable from data with event/report dates ≤ end of month `t`. Document any source where only a "current snapshot" exists (e.g. NDC marketing status) as a known limitation.
4. **Demo-mode fallback.** Every pipeline stage reads/writes cached parquet under `data/`. The dashboard and all analysis must run fully offline from those caches. Commit small derived parquets if license-safe; raw dumps stay gitignored.
5. **Verification without Chrome.** The dashboard is **Plotly Dash**. Do NOT use `dash[testing]` (it requires Selenium/Chrome). Instead: unit-test callback functions directly as plain functions, and smoke-test the served app with Flask's test client (`app.server.test_client().get('/')` → 200, plus a request to the layout/dependencies endpoints).

---

## Repo layout

```
drug-shortage-radar/
├── PLAN.md                  # this file
├── README.md                # written in Phase 8
├── requirements.txt
├── config.py                # paths, date windows, panel definitions, split dates
├── data/
│   ├── raw/                 # gitignored API dumps (json/csv)
│   └── processed/           # parquet: panel, features, labels, predictions
├── src/
│   ├── ingest/
│   │   ├── shortages.py     # openFDA drug shortages
│   │   ├── ndc.py           # openFDA NDC directory
│   │   ├── drugsfda.py      # Drugs@FDA applications
│   │   ├── inspections.py   # FDA Data Dashboard: classifications + citations
│   │   ├── compliance.py    # FDA Data Dashboard: compliance actions (incl. warning letters)
│   │   ├── enforcement.py   # openFDA recall/enforcement
│   │   └── nadac.py         # CMS NADAC drug prices
│   ├── resolve.py           # entity resolution: firms ↔ drugs
│   ├── panel.py             # drug-month panel + labels
│   ├── features.py          # feature engineering (leakage-safe)
│   ├── models.py            # baselines, logistic, LightGBM, calibration
│   ├── evaluate.py          # validation metrics, backtest, lead-time analysis
│   └── explain.py           # SHAP / driver breakdown for dashboard
├── scripts/
│   ├── run_ingest.py        # pull all sources → data/raw + normalized parquet
│   ├── build_panel.py       # panel + features + labels → data/processed
│   ├── train.py             # train + validate (NEVER touches test window)
│   └── eval_test_LOCKED.py  # final test eval — only run when user says so
├── app.py                   # Dash dashboard
└── tests/
    ├── test_leakage.py      # asserts no feature uses post-t data
    ├── test_panel.py        # label correctness on hand-checked examples
    └── test_app.py          # callback unit tests + Flask test-client smoke test
```

`requirements.txt`: `pandas pyarrow requests lightgbm scikit-learn lifelines shap dash plotly rapidfuzz pytest tenacity beautifulsoup4 lxml`.

---

## Phase 1 — Data acquisition (`src/ingest/`)

Each ingester: pull → save raw to `data/raw/` → normalize to typed parquet in `data/processed/`. Make every ingester idempotent and resumable (skip if raw file exists unless `--force`). Use `tenacity` for retries. Respect openFDA paging (max `limit=1000` per call for most endpoints, `skip` capped at 25k — use the **bulk download files** from https://open.fda.gov/data/downloads/ for full datasets instead of paging where available).

### 1a. Drug shortages (labels) — VERIFIED 2026-06-10, read carefully

**Verified finding: the live openFDA shortages endpoint is a snapshot, NOT an archive.** Probed `https://api.fda.gov/drug/shortages.json` directly: 1,681 rows (one per package-NDC), only 313 distinct (generic_name, initial_posting_date) events, status breakdown Current=1,147 / To Be Discontinued=505 / **Resolved=29**. Resolved shortages are purged from the dataset (e.g. only 26 rows have a 2024 posting date). FDA's own CSV export (`https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm`, 1,682 rows) is the identical snapshot. **Neither can supply historical labels alone.**

**Label construction therefore uses Wayback Machine snapshot reconstruction** (verified available via the CDX API, `http://web.archive.org/cdx/search/cdx?url=...&fl=timestamp&filter=statuscode:200`):
- CSV snapshots of `accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm`: 96 captures, 2019-10 → present (thin in 2022: 2 captures).
- HTML snapshots of `accessdata.fda.gov/scripts/drugshortages/default.cfm`: ~2,400 captures across 127 distinct months back to 2014, dense 2020→present (e.g. 302 in 2022, 590 in 2023). The page embeds the full current/resolved/discontinuation lists (~200KB).

Procedure (`src/ingest/shortages.py`):
1. Pull CDX snapshot lists for both URLs; collapse to ~2 snapshots/month; download via `https://web.archive.org/web/{timestamp}id_/{original_url}` at ≤1 req/sec with retries; cache everything in `data/raw/wayback/` (idempotent — never re-download).
2. Parse CSV snapshots first (simplest, 2019-10+). Then HTML snapshots for pre-2019 history and 2022 gap-fill — expect format drift across eras; write per-era parsers and time-box HTML parsing to ~2 days, prioritizing 2018+.
3. Union records keyed by (normalized generic_name, presentation, company). **Onset = `Initial Posting Date`** (carried in every snapshot, so onsets are exact even with sparse captures). Resolution ≈ first snapshot where the record shows Resolved status or disappears (interval-censored; store the bounding snapshot dates).
4. Also pull the live openFDA endpoint each run: its `openfda` enrichment block (present on 1,499/1,681 rows: application_number, product_ndc, route, manufacturer_name) is valuable for the Phase 2 crosswalk, and the live feed powers the dashboard's "current status" display.

Parsing gotchas (verified): dates are `MM/DD/YYYY` strings; CSV headers have stray leading/trailing spaces (strip all keys); availability values contain the typo `Avaliable`; one row per package-NDC so dedupe to event level.

**Known label limitation (state it in README):** shortages that began AND resolved entirely within a snapshot gap are missed — material mainly before 2020. Quantify by comparing reconstructed onset counts per year against published counts (e.g. HHS ASPE drug-shortage analyses). Set `config.STUDY_START = 2018-01` if HTML parsing succeeds, else `2020-01`.

### 1b. Drug universe — openFDA NDC directory + Drugs@FDA
- NDC directory (`https://api.fda.gov/drug/ndc.json`, bulk download available): generic name, brand name, labeler, dosage form, route, DEA schedule, marketing category (ANDA/NDA/OTC), product NDC, marketing start/end dates.
- Drugs@FDA (`https://api.fda.gov/drug/drugsfda.json`, bulk available): application number, sponsor, approval dates — gives drug age and sponsor identity.
- Define the **drug unit**: `drug_key = normalized(generic_name) + route` (e.g. `vancomycin|injection`). This matches the granularity of the shortage database. Build a crosswalk table: drug_key ↔ NDCs ↔ labelers ↔ application numbers ↔ sponsors.
- Restrict universe to prescription products (marketing category NDA/ANDA/BLA) to keep the panel meaningful; record universe size (expect 2–4k drug_keys).

### 1c. Inspections & citations — FDA Data Dashboard
- Dashboard: https://datadashboard.fda.gov/oii/cd/inspections.htm — try full-dataset Excel/CSV export first (no key). API (needs free key from the OII Unified Logon app): documented at https://datadashboard.fda.gov/oii/api/index.htm with field definitions for Inspections Classifications and Inspections Citations.
- Keep (inspections): FEI number, firm name, address/country, inspection end date, classification (NAI/VAI/OAI), project area (filter to Drugs).
- Keep (citations): FEI, inspection date, CFR citation number + short description (text used in Phase 6).
- If neither export nor key is workable within a day of effort, fallback: FDA's Inspection Classification Database download at https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/inspection-classification-database

### 1d. Compliance actions — FDA Data Dashboard
- Same dashboard family; includes warning letters, injunctions, seizures with firm name/FEI and action date. Field docs: https://datadashboard.fda.gov/oii/api/api-definitions-compliance-actions.htm

### 1e. Recalls — openFDA enforcement
- `https://api.fda.gov/drug/enforcement.json` (bulk available): recalling firm, product description, classification (I/II/III), recall initiation date, reason. Product description text is matched to drug_keys in Phase 2.

### 1f. Prices — CMS NADAC
- NADAC (National Average Drug Acquisition Cost) weekly files from data.medicaid.gov (free CSV API, no key). Keep: NDC, unit price, effective date.
- Why: chronically low/falling generic prices are a known shortage driver (thin margins → manufacturer exit). Compute per-drug_key median price and 12-month price trend.

### 1g. Stretch only (skip unless ahead of schedule)
- DEA quota notices via Federal Register API (controlled substances), FAERS adverse events. Do not block on these.

**Phase 1 acceptance:** `python scripts/run_ingest.py` completes; each source has a parquet in `data/processed/` with a row count and date-coverage printout; reconstructed shortage-onset counts per year documented and sanity-checked against published figures.

---

## Phase 2 — Entity resolution (`src/resolve.py`)

The grind. Inspections/compliance/recalls are keyed by **firm** (FEI + name); drugs are keyed by **labeler/sponsor name**. Build `firm_link` table: (FEI/firm_name) → (labeler/sponsor) → drug_keys.

1. Normalize names aggressively (uppercase, strip punctuation and corporate suffixes: INC, LLC, CORP, LTD, PHARMACEUTICALS/PHARMA/PHARM, LABORATORIES/LABS, USA, CO).
2. Exact match on normalized names first; then `rapidfuzz` token-set ratio ≥ 92 for fuzzy candidates.
3. **Manual crosswalk for the top ~50 firms** by inspection count (subsidiary names: e.g. Hospira→Pfizer, Sandoz↔Novartis). Store as a reviewed CSV in `src/` so it's versioned and auditable.
4. Recall product descriptions → drug_keys: match generic name tokens against the drug universe vocabulary.
5. **Quantify match quality:** report % of drug-inspections linked, and spot-check 20 random links by hand; record precision estimate in README. Imperfect linkage is acceptable and must be discussed honestly in the writeup — it attenuates signal, it doesn't invalidate the design.

**Acceptance:** `firm_link` parquet exists; ≥60% of FDA drug-project inspections link to at least one drug_key; spot-check documented.

---

## Phase 3 — Panel, labels, splits (`src/panel.py`)

- **Panel:** one row per (drug_key, month) from `config.STUDY_START` to present. Drug enters panel when its first NDC is marketed, exits when all NDCs end marketing.
- **Label:** `y = 1` if a shortage with matching drug_key has `initial_posting_date` within `(t, t + 6 months]`, else 0. Rows where the drug is **already in an active shortage at t are excluded** from training/eval (we predict onsets, not persistence).
- Also store onset month for lead-time analysis and a 3-month label variant for sensitivity analysis.
- **Splits (set in `config.py`; boundaries assume the verified label window of ~2018/2020 → present — adjust only if reconstruction lands differently, keep the structure):**
  - Train: STUDY_START → 2022-12
  - Validation: 2023-01 → 2024-12 (all iteration happens here)
  - **Test (LOCKED): 2025-01 → latest month with a complete 6-month label horizon (≈ 2025-11 as of June 2026).** Only `scripts/eval_test_LOCKED.py` may read it, and it must print a loud confirmation prompt before running.
- `tests/test_panel.py`: hand-verify labels for 3 known shortages (e.g. amoxicillin 2022, Adderall/amphetamine salts 2022, cisplatin/carboplatin 2023) — the panel must show y=1 in the months preceding the posting date.

**Acceptance:** panel parquet with printed shape, base rate per split (expect ~1–3% positive), and the 3 hand-checks passing.

---

## Phase 4 — Features (`src/features.py`)

All features computed **as-of month t** (event date ≤ t). Group by theme; suffix windows like `_12m`, `_36m`.

**Market structure (from NDC/Drugs@FDA):**
- `n_active_labelers`, `n_active_anda_holders` — the single most cited shortage driver is few-supplier markets
- `labeler_hhi` (concentration index), `is_generic` (ANDA share), `drug_age_years`
- `route_injectable` flag (sterile injectables dominate shortages), dosage form one-hots, `dea_scheduled` flag

**Manufacturing risk (via firm_link):**
- `n_oai_inspections_12m/_36m`, `n_vai_12m` at linked firms; `worst_classification_24m`
- `n_citations_12m`, plus counts for citation themes (sterility/aseptic, data integrity, equipment) keyword-matched from citation descriptions
- `warning_letters_24m`, `compliance_actions_36m`
- `share_of_suppliers_with_oai_24m` — fraction of this drug's labelers under recent OAI

**Recalls:** `recalls_12m` (drug-level), `recalls_class1_24m`, `firm_recalls_12m` (any product at linked firms)

**Price pressure (NADAC):** `nadac_price_trend_12m` (slope of log price), `nadac_price_level_pctile` within dosage-form peers

**History:** `past_shortages_count` (this drug, before t), `months_since_last_shortage`, `class_shortage_rate_12m` (shortage frequency in the drug's ATC-like class — approximate class via first word of generic name or route+form peer group)

`tests/test_leakage.py`: for a sampled (drug, t), recompute each feature with data truncated at t and assert equality with the panel value; plus a unit test that a synthetic future-dated event does not move any feature at t.

**Acceptance:** feature matrix parquet; leakage tests pass; feature null rates printed.

---

## Phase 5 — Models (`src/models.py`, `scripts/train.py`)

In strict order — each must beat the previous on validation before moving on:

1. **Naive baselines** (the bar to clear): (a) predict by `past_shortages_count`; (b) injectable + few-suppliers rule.
2. **Logistic regression** (standardized features, class weights) — interpretable reference.
3. **LightGBM** (primary model): `scale_pos_weight` or class weights for imbalance, early stopping on validation PR-AUC, modest grid (depth, leaves, learning rate, min_child_samples). No SMOTE — weights only.
4. **Calibration:** isotonic on a temporal slice of train held out from fitting; report Brier + reliability table.
5. **Stretch (only if validation results are solid):** Cox proportional hazards (`lifelines`) framing time-to-shortage, as a writeup-enriching comparison.

Evaluation on **validation only** (`src/evaluate.py`):
- PR-AUC (primary; ROC-AUC secondary)
- **Recall@50 and Precision@50** per month — "if a pharmacist watched our top-50 list, what fraction of real shortages would they have seen coming?" This is the headline metric.
- **Lead time:** for true positives, months between first month the drug entered top-50 and shortage posting
- Ablations: drop manufacturing-risk features / price features / market-structure features → quantify each theme's contribution
- Rolling-origin backtest within train+validation (train ≤ t, predict t+1..t+6, slide quarterly) to show temporal stability

**Acceptance:** a `reports/validation.md` (auto-generated) with all metrics + ablation table; LightGBM beats both baselines on PR-AUC and Recall@50.

## Phase 6 — NLP signal (enhancement, time-boxed to ~1 day)

Embed citation short-descriptions + shortage reason texts with `sentence-transformers` (`all-MiniLM-L6-v2`, local, no API). Cluster citation embeddings (HDBSCAN or k-means); add per-drug counts of risk-cluster citations in the last 24m as features. Keep only if validation PR-AUC improves; otherwise report as a negative result (fine for a capstone). If `sentence-transformers` install is heavy, TF-IDF + SVD is an acceptable substitute.

---

## Phase 7 — Dash dashboard (`app.py`)

Plotly **Dash** app. Reads only from `data/processed/` parquet (works fully offline = demo mode). Load parquet once at startup into module-level dataframes; keep callbacks pure functions over those frames (this is also what makes them unit-testable).

Layout: `dcc.Tabs` with three tabs.
- **Watchlist tab:** `dash_table.DataTable` of top-N at-risk drugs not currently in shortage, ranked by calibrated probability; dropdown filters for route/dosage form/therapeutic category; conditional row styling by risk band; badge column for drugs that later did go short (non-test windows only).
- **Drug detail tab:** drug selector (`dcc.Dropdown`) → risk-score history sparkline (plotly), SHAP driver breakdown bar chart ("3 of 4 suppliers had OAI inspections in the last 18 months; price fell 40%"), timeline of linked inspections/recalls/compliance events, past shortage history, and current live status from the openFDA feed cache.
- **Model card tab:** validation metrics table, lead-time histogram, honest limitations (entity-resolution noise, snapshot-only NDC status, label = FDA posting date not true onset, Wayback gap censoring).

Structure callbacks as `update_watchlist(filters...) -> data`, `update_drug_detail(drug_key) -> figures` so tests can call them directly.

`tests/test_app.py` (no Chrome/Selenium — do not install `dash[testing]`):
1. Import `app`; assert layout builds and key component IDs exist.
2. Call each callback function directly with sample inputs; assert non-empty outputs.
3. `app.server.test_client().get('/')` returns 200 and the Dash index HTML; `/_dash-layout` and `/_dash-dependencies` return 200.

**Acceptance:** `python app.py` serves on localhost from caches alone; `pytest tests/test_app.py` green.

## Phase 8 — Writeup + final test evaluation

1. README: problem, data lineage diagram, label/leakage design, entity-resolution quality, validation results, limitations, ethics note (false alarms can themselves distort purchasing — frame as triage, not oracle).
2. **Only when the user explicitly asks:** run `scripts/eval_test_LOCKED.py` once on the 2024+ window with the final frozen model; append results to README verbatim, good or bad.

---

## Execution notes for the implementing agent

- Work the phases in order; each has an acceptance gate — print/verify it before moving on. Commit at each phase boundary.
- When a source fights back (export format changed, endpoint paging quirks), spend ≤2h, then use the listed fallback and note the substitution in README. Do not silently drop a source.
- Expected overall scale is small: panel ≈ a few hundred thousand rows, features < 100 columns. Everything runs locally on CPU; no cloud, no GPU.
- The openFDA-snapshot problem is already solved by design (Phase 1a Wayback reconstruction — verified 2026-06-10). If Wayback reconstruction itself underdelivers (e.g. <150 distinct onsets in 2020–2024), that is a stop-and-report moment, not a reason to silently shrink scope.
- Be polite to web.archive.org: ≤1 req/sec, exponential backoff on 429/503, cache every download. The full pull is a few hundred files — run it once, then work from cache.
- Surprises that change the design (label base rate <0.3%, linkage <40%, NADAC join failures) are user-decision points: stop and report rather than improvising around them.
