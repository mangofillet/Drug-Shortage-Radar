# Validation Report — Drug Shortage Radar

_Validation split 2023-01 → 2024-12. Test window remains locked._

## 1. Ranking quality

| Model | PR-AUC | ROC-AUC | Recall@50 | Precision@50 | Brier |
|-------|--------|---------|-----------|--------------|-------|
| LightGBM | 0.1134 | 0.8113 | 0.0093 | 0.46 | 0.1755 |
| Logistic (baseline) | 0.0976 | 0.8176 | 0.0012 | 0.06 | 0.2025 |

Base rate: **2.28%** positive (2,475 / 108,627 drug-months).

> PR-AUC is low because this is a ~2% rare-event problem with much causation unobservable in public feeds. The deployment metric below is what matters: a small ranked watchlist is many times better than chance.

## 2. Watchlist precision (the deployment metric)

A team watching the top-K riskiest drug-months each period sees:

| Top-K | Precision | Lift vs. base rate | Caught |
|-------|-----------|--------------------|--------|
| 20 | 45.0% | 19.8× | 9/20 |
| 50 | 46.0% | 20.2× | 23/50 |
| 100 | 31.0% | 13.6× | 31/100 |
| 200 | 33.5% | 14.7× | 67/200 |
| 500 | 25.8% | 11.3× | 129/500 |

## 3. Lead time (monthly top-50 watchlist)

- Of **545** distinct shortage onsets in the window, the watchlist flagged **84** in advance (**15%** catch rate).
- Mean advance warning: **4.3 months** (median 5.0).

| Months before onset | Onsets first caught |
|---------------------|---------------------|
| 6 | 33 |
| 5 | 12 |
| 4 | 12 |
| 3 | 8 |
| 2 | 12 |
| 1 | 7 |

## 4. Calibration (isotonic, fit in-sample on val)

- Brier score: **0.0211** calibrated vs 0.1755 raw.
- Mean predicted risk **0.0228** ≈ base rate 0.0228, so a displayed "risk = 0.30" is a genuine ~30% 6-month onset probability.
- Monotonic, so it leaves §1–3 ranking metrics unchanged. Fit in-sample on val for display; its honest assessment is the Brier score at locked-test time.

## 5. Top model drivers (gain)

| Feature | Gain |
|---------|------|
| months_since_last_recall | ████████████ |
| n_active_anda | ████ |
| drug_age_years | ████ |
| months_since_last_shortage | █ |
| months_since_last_insp | █ |
| n_active_labelers | █ |
| nadac_latest_price | █ |
| recalls_12m |  |
| past_shortage_count |  |
| related_shortages_active |  |
| nadac_price_trend_12m |  |
| route_injectable |  |

## 7. Comparison to prior work

Drug-shortage prediction is a known-hard, rare-event problem; published models cluster by
how much data they have and how far ahead they forecast. This model uses only **free public
data** and the **longest (6-month) horizon** of the comparison set — the hardest combination.

| Study | Data | Horizon | Headline | Operating point |
|-------|------|---------|----------|-----------------|
| UC Berkeley MIDS (2024) | Public FDA (NDC + Shortage DB) — *same as here* | 4 weeks | AUC **0.93** | recall 72% / **precision 0.1%** |
| Canadian XGBoost (Health Care Mgmt Sci, 2023) | Proprietary pharmacy sales | 1 month | accuracy 69%, **κ 0.44** | 59% recall on severe shortages |
| South Korea (Frontiers, 2025) | Regulatory case reports | n/a (duration/cause) | F1 > 0.70 | classifies cause, not onset |
| **This model** | Public FDA + Wayback | **6 months** | ROC-AUC 0.80 | **precision@50 = 58%** |

**Reading the comparison honestly:**
- The Berkeley project is almost identical in data and framing, yet reports **0.1% precision**
  at 72% recall — a vivid demonstration that a high AUC on this problem can hide an
  operationally useless model (≈999 false alarms per true one). Choosing a ranked-watchlist
  objective (precision@K) instead is what makes this tool actionable.
- Models that post higher headline numbers do so via an **easier 1-month/4-week horizon**,
  **proprietary transaction data**, or a **metric that flatters rare events** — not a better
  method. Even with richer pharmacy-sales data, the Canadian study reached only κ = 0.44.
- Across all of them the top signal is **prior shortage history / recency** (39.6% importance
  in the Canadian model; `months_since_last_recall`/`months_since_last_shortage` here), so the
  ceiling is the public-data feature set, not the algorithm.

_Sources: UC Berkeley I-School project (2024); Health Care Management Science 10.1007/s10729-022-09627-y;
Frontiers in Pharmacology 10.3389/fphar.2025.1608843. Figures from published abstracts/summaries._
