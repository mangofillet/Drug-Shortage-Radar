"""Phase 7 — SHAP driver breakdowns for dashboard drug-detail tab."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

MODELS_DIR = Path(__file__).parent.parent / "data" / "processed" / "models"


def compute_shap(features: pd.DataFrame, drug_key: str | None = None) -> pd.DataFrame:
    """
    Compute SHAP values for the LightGBM model.
    If drug_key is given, returns only that drug's values.
    Returns DataFrame with columns: drug_key, month, feature, shap_value, feature_value.
    """
    lgb_model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_final.txt"))
    with open(MODELS_DIR / "feature_cols.json") as f:
        feat_cols = json.load(f)

    df = features.copy()
    if drug_key:
        df = df[df["drug_key"] == drug_key]

    cols = [c for c in feat_cols if c in df.columns]
    X = df[cols].fillna(-1).values.astype(np.float32)

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X)
    # Some SHAP versions return a list [neg_class, pos_class] for binary classifiers;
    # take the positive-class contributions.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        for j, feat in enumerate(cols):
            rows.append({
                "drug_key": row["drug_key"],
                "month": row["month"],
                "feature": feat,
                "shap_value": float(shap_values[i, j]),
                "feature_value": float(X[i, j]),
            })
    return pd.DataFrame(rows)


def top_drivers(features: pd.DataFrame, drug_key: str, month: pd.Timestamp, top_n: int = 5) -> list[dict]:
    """Return top N SHAP drivers for a specific drug at a specific month."""
    shap_df = compute_shap(features, drug_key=drug_key)
    row = shap_df[shap_df["month"] == month]
    if len(row) == 0:
        return []
    row = row.reindex(row["shap_value"].abs().sort_values(ascending=False).index)
    return row.head(top_n)[["feature", "shap_value", "feature_value"]].to_dict("records")
