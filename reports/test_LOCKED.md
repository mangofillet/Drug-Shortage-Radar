# LOCKED Test Evaluation — Drug Shortage Radar

_Test window 2025-01 → 2025-11. Run ONCE; never used for tuning. Validation shown alongside for reference._

## Headline (val vs test)

| Metric | Val (2023–24) | **Test (2025)** |
|--------|---------------|-----------------|
| PR-AUC | 0.1134 | **0.107** |
| ROC-AUC | 0.8113 | **0.8292** |
| Precision@50 | 0.46 | **0.24** |
| Brier (calibrated) | 0.0211 | **0.0202** |
| Base rate | 0.0228 | 0.0216 |
| Lead time mean (mo) | 4.3 | **4.29** |
| Onsets caught (top-50/mo) | 84/545 | **31/274** |

## Test watchlist precision@K

| Top-K | Precision | Lift | Caught |
|-------|-----------|------|--------|
| 20 | 30.0% | 13.9× | 6/20 |
| 50 | 24.0% | 11.1× | 12/50 |
| 100 | 19.0% | 8.8× | 19/100 |
| 200 | 18.0% | 8.3× | 36/200 |
| 500 | 18.6% | 8.6× | 93/500 |

Test rows: 54,592 (1181 positive).
