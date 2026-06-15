"""
Phase 6 — build the validation report from the trained model.

Reads the scored predictions + saved artifacts (no retraining), and writes
reports/validation.md with: ranking metrics, watchlist precision@K, lead-time
distribution, and top feature drivers. Validation split only.
Usage: python scripts/evaluate.py
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import DATA_PROCESSED
from src.models import FEATURE_COLS, save_predictions
from src.evaluate import (
    evaluate_model, precision_at_k_curve, lead_time_analysis, write_validation_report,
)

MODELS_DIR = DATA_PROCESSED / "models"


def _logistic_val_scores(val: pd.DataFrame) -> np.ndarray | None:
    path = MODELS_DIR / "logistic.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    medians = pd.Series(bundle["medians"])
    X = val[FEATURE_COLS].fillna(medians)
    X_s = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(X_s)[:, 1]


def main():
    feat_path = DATA_PROCESSED / "features_with_labels.parquet"
    pred_path = DATA_PROCESSED / "predictions_val.parquet"
    if not feat_path.exists():
        print("ERROR: Run scripts/build_panel.py then scripts/train.py first")
        sys.exit(1)

    features = pd.read_parquet(feat_path)
    if not pred_path.exists():
        print("Predictions missing — scoring now…")
        save_predictions(features)
    preds = pd.read_parquet(pred_path)

    val_preds = preds[preds["split"] == "val"].copy()
    y = val_preds["y_6m"].astype(int).values
    # Rank on the RAW model output (isotonic calibration introduces ties that would
    # perturb ranking metrics); calibrated score is only for displayed-probability Brier.
    rank_col = "risk_score_raw" if "risk_score_raw" in val_preds.columns else "risk_score"
    lgbm_scores = val_preds[rank_col].values
    lead_preds = val_preds.copy()
    lead_preds["risk_score"] = lead_preds[rank_col]  # rank lead-time on raw scores too

    metrics = [evaluate_model("LightGBM", lgbm_scores, y)]

    val_feat = features[features["split"] == "val"]
    lr_scores = _logistic_val_scores(val_feat)
    if lr_scores is not None:
        metrics.append(evaluate_model("Logistic (baseline)", lr_scores, val_feat["y_6m"].astype(int).values))

    pk_curve = precision_at_k_curve(y, lgbm_scores, ks=(20, 50, 100, 200, 500))
    lead = lead_time_analysis(lead_preds, k=50)

    # Calibration quality (isotonic, in-sample on val) — display/Brier only.
    from sklearn.metrics import brier_score_loss
    calibration = None
    if "risk_score_raw" in val_preds.columns:
        calibration = {
            "brier_raw": round(brier_score_loss(y, val_preds["risk_score_raw"].clip(0, 1)), 4),
            "brier_calibrated": round(brier_score_loss(y, val_preds["risk_score"]), 4),
            "mean_pred": round(float(val_preds["risk_score"].mean()), 4),
            "base_rate": round(float(y.mean()), 4),
        }

    imp_path = MODELS_DIR / "feature_importance.parquet"
    importance = pd.read_parquet(imp_path) if imp_path.exists() else None

    report = write_validation_report(metrics, pk_curve, lead, importance=importance, calibration=calibration)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for m in metrics:
        print(f"  {m['model']:22s} PR-AUC={m['pr_auc']}  ROC-AUC={m['roc_auc']}  P@50={m['precision_at_50']}")
    if pk_curve:
        top = pk_curve[1] if len(pk_curve) > 1 else pk_curve[0]
        print(f"  Watchlist precision@{top['k']}: {top['precision']*100:.1f}% ({top['lift']}× lift)")
    if lead.get("n_onsets_caught", 0):
        print(f"  Lead time: caught {lead['n_onsets_caught']}/{lead['n_onsets_total']} onsets, "
              f"mean {lead['mean_lead_months']} mo advance warning")
    print("\nReport → reports/validation.md")


if __name__ == "__main__":
    main()
