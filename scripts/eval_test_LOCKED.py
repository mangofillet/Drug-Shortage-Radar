"""
LOCKED TEST SET EVALUATION — DO NOT RUN UNLESS USER EXPLICITLY REQUESTS IT.

Evaluates the final frozen model on the 2025-01 → 2025-11 test window, ONCE, at the
user's explicit request. Running it prematurely (or more than once while iterating)
invalidates the capstone's honest out-of-sample estimate.

Mirrors the validation metrics exactly (src/evaluate.py) so val↔test is apples-to-apples.
Writes reports/test_LOCKED.md. Confirmation gate preserved: pipe the phrase via stdin.
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from config import DATA_PROCESSED, REPORTS
from src.models import FEATURE_COLS, load_final_model
from src.evaluate import evaluate_model, precision_at_k_curve, lead_time_analysis

CONFIRMATION = "I confirm I want to run the locked test evaluation"


def _score(df: pd.DataFrame, split: str):
    """Return (y, raw_scores, calibrated_scores) for one split."""
    booster = load_final_model()
    d = df[df["split"] == split].copy()
    d["risk_score_raw"] = booster.predict(d[FEATURE_COLS])
    with open(DATA_PROCESSED / "models" / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)
    d["risk_score_cal"] = iso.predict(d["risk_score_raw"].values)
    return d


def run():
    features = pd.read_parquet(DATA_PROCESSED / "features_with_labels.parquet")

    results = {}
    for split in ["val", "test"]:
        d = _score(features, split)
        y = d["y_6m"].astype(int).values
        raw = d["risk_score_raw"].values
        m = evaluate_model(f"LightGBM ({split})", raw, y)
        pk = precision_at_k_curve(y, raw, ks=(20, 50, 100, 200, 500))
        lead_df = d.copy()
        lead_df["risk_score"] = lead_df["risk_score_raw"]
        lead = lead_time_analysis(lead_df, k=50)
        results[split] = {
            "m": m, "pk": pk, "lead": lead,
            "brier_raw": round(brier_score_loss(y, np.clip(raw, 0, 1)), 4),
            "brier_cal": round(brier_score_loss(y, d["risk_score_cal"]), 4),
            "base": round(float(y.mean()), 4),
        }

    v, t = results["val"], results["test"]

    def p50(r):
        return next((x["precision"] for x in r["pk"] if x["k"] == 50), None)

    print("\n" + "=" * 64)
    print("LOCKED TEST EVALUATION — 2025-01 → 2025-11  (val shown for reference)")
    print("=" * 64)
    print(f"{'metric':22s}{'VAL':>12s}{'TEST':>12s}")
    print("-" * 46)
    print(f"{'PR-AUC':22s}{v['m']['pr_auc']:>12}{t['m']['pr_auc']:>12}")
    print(f"{'ROC-AUC':22s}{v['m']['roc_auc']:>12}{t['m']['roc_auc']:>12}")
    print(f"{'precision@50':22s}{p50(v):>12}{p50(t):>12}")
    print(f"{'Brier (calibrated)':22s}{v['brier_cal']:>12}{t['brier_cal']:>12}")
    print(f"{'base rate':22s}{v['base']:>12}{t['base']:>12}")
    print(f"{'lead mean (mo)':22s}{v['lead'].get('mean_lead_months',0):>12}{t['lead'].get('mean_lead_months',0):>12}")
    print(f"{'onsets caught':22s}{str(v['lead'].get('n_onsets_caught'))+'/'+str(v['lead'].get('n_onsets_total')):>12}"
          f"{str(t['lead'].get('n_onsets_caught'))+'/'+str(t['lead'].get('n_onsets_total')):>12}")

    # ── Write report ──
    L = ["# LOCKED Test Evaluation — Drug Shortage Radar\n",
         "_Test window 2025-01 → 2025-11. Run ONCE; never used for tuning. "
         "Validation shown alongside for reference._\n",
         "## Headline (val vs test)\n",
         "| Metric | Val (2023–24) | **Test (2025)** |",
         "|--------|---------------|-----------------|",
         f"| PR-AUC | {v['m']['pr_auc']} | **{t['m']['pr_auc']}** |",
         f"| ROC-AUC | {v['m']['roc_auc']} | **{t['m']['roc_auc']}** |",
         f"| Precision@50 | {p50(v)} | **{p50(t)}** |",
         f"| Brier (calibrated) | {v['brier_cal']} | **{t['brier_cal']}** |",
         f"| Base rate | {v['base']} | {t['base']} |",
         f"| Lead time mean (mo) | {v['lead'].get('mean_lead_months')} | **{t['lead'].get('mean_lead_months')}** |",
         f"| Onsets caught (top-50/mo) | {v['lead'].get('n_onsets_caught')}/{v['lead'].get('n_onsets_total')} "
         f"| **{t['lead'].get('n_onsets_caught')}/{t['lead'].get('n_onsets_total')}** |",
         "\n## Test watchlist precision@K\n",
         "| Top-K | Precision | Lift | Caught |",
         "|-------|-----------|------|--------|"]
    for r in t["pk"]:
        L.append(f"| {r['k']} | {r['precision']*100:.1f}% | {r['lift']}× | {r['n_caught']}/{r['k']} |")
    L.append(f"\nTest rows: {t['m']['n_total']:,} ({t['m']['n_positive']} positive).\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "test_LOCKED.md"
    path.write_text("\n".join(L))
    print(f"\nReport written → {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("WARNING: This runs the LOCKED test set evaluation.")
    print("This should only be done once, at the user's explicit request.")
    print("=" * 70)
    response = input(f'Type exactly: "{CONFIRMATION}"\n> ')
    if response.strip() != CONFIRMATION:
        print("Aborted.")
        sys.exit(1)
    print("Proceeding with test evaluation...\n")
    run()
