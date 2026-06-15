"""
Phase 5 — Model training with time-series cross-validation.

Workflow:
  1. Logistic regression baseline (train → val)
  2. LightGBM with 3-fold forward-chaining CV on train set to tune n_estimators
  3. Refit final LightGBM on all train data with early stopping on val
  4. Val PR-AUC reported as headline metric
  5. Test set NEVER touched here — only via scripts/eval_test_LOCKED.py
"""

import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import DATA_PROCESSED

MODELS_DIR = DATA_PROCESSED / "models"
FEATURE_COLS = [
    "n_active_labelers", "n_active_anda", "is_generic", "labeler_hhi",
    "route_injectable", "dea_scheduled", "drug_age_years",
    "n_oai_12m", "n_oai_36m", "n_vai_12m", "n_insp_12m", "n_insp_24m",
    "worst_class_24m", "months_since_last_insp",
    "recalls_6m", "recalls_12m", "recalls_class1_24m", "months_since_last_recall",
    "nadac_latest_price", "nadac_price_trend_12m",
    "past_shortage_count", "months_since_last_shortage",
    "firm_ever_compliance_action",
    "related_shortages_active",
    # ── Excluded after investigation (computed in features.py, left out here): ──
    # n_citations_24m: openFDA citations source has a 2011-2022 gap (data only for
    #   2008-2010 and 2023+), so it is zero across the entire train window but nonzero
    #   in val/test — a train/serve skew that can only hurt.
    # labelers_exited_12m / net_labeler_change_12m / months_since_last_exit: the openFDA
    #   NDC directory is a CURRENT snapshot — discontinued products are purged and every
    #   non-null marketing_end_date is future-dated, so historical supplier exits are
    #   unobservable (labelers_exited_12m == 0 everywhere, months_since_last_exit 100%
    #   null). Building this would require archived NDC snapshots, not the live bulk file.
]
LABEL_COL = "y_6m"

LGBM_BASE = dict(
    objective="binary",
    metric="average_precision",
    n_estimators=1000,
    learning_rate=0.05,
    # Shallower trees + larger leaves = stronger regularization. With only ~5k positives
    # the previous num_leaves=31 / min_child_samples=50 overfit (train PR-AUC 0.20 vs val
    # 0.11); these values narrow that gap at no cost to the precision@K watchlist.
    num_leaves=16,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.1,
    verbose=-1,
    n_jobs=-1,
    random_state=42,
)

# Forward-chaining CV folds within the train window (2018-01 to 2022-12)
CV_FOLDS = [
    ("2019-12", "2020-01", "2020-12"),
    ("2020-12", "2021-01", "2021-12"),
    ("2021-12", "2022-01", "2022-12"),
]


def _slice(features: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    mask = pd.Series(True, index=features.index)
    if start:
        mask &= features["month"] >= pd.Timestamp(start)
    if end:
        mask &= features["month"] <= pd.Timestamp(end)
    return features[mask].copy()


def _xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)
    return X, y


def _metrics(y_true, y_prob, label: str = "") -> dict:
    pr_auc = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    k = max(1, int(y_true.sum()))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    prec_at_k = float(np.array(y_true)[top_k_idx].mean())
    if label:
        print(f"    {label:35s}  PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}  Prec@K={prec_at_k:.3f}  (K={k})")
    return {"pr_auc": pr_auc, "roc_auc": roc, "prec_at_k": prec_at_k, "k": k}


def train_logistic(features: pd.DataFrame) -> dict:
    print("\n── Logistic Regression Baseline ──")
    train = features[features["split"] == "train"]
    val   = features[features["split"] == "val"]

    X_tr, y_tr   = _xy(train)
    X_val, y_val = _xy(val)

    medians = X_tr.median()
    X_tr  = X_tr.fillna(medians)
    X_val = X_val.fillna(medians)

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(
        max_iter=1000, C=0.1, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_tr_s, y_tr)
    prob_val = clf.predict_proba(X_val_s)[:, 1]

    m = _metrics(y_val, prob_val, "Logistic (val 2023-2024)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "logistic.pkl", "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "medians": medians.to_dict()}, f)
    print(f"  Saved → {MODELS_DIR}/logistic.pkl")
    return m


def train_lgbm_cv(features: pd.DataFrame) -> dict:
    print("\n── LightGBM Time-Series CV (3 folds) ──")
    train_all = features[features["split"] == "train"].copy()
    fold_results = []

    for i, (tr_end, ev_start, ev_end) in enumerate(
        tqdm(CV_FOLDS, desc="CV folds", unit="fold", ncols=72)
    ):
        print(f"\n  Fold {i+1}/3  train→{tr_end}  eval {ev_start}→{ev_end}")
        tr = _slice(train_all, end=tr_end)
        ev = _slice(train_all, start=ev_start, end=ev_end)

        if len(tr) == 0 or len(ev) == 0:
            print("    skipped (empty split)")
            continue

        X_tr, y_tr = _xy(tr)
        X_ev, y_ev = _xy(ev)

        spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
        model = lgb.LGBMClassifier(**{**LGBM_BASE, "scale_pos_weight": spw})
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_ev, y_ev)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )

        best_iter = model.best_iteration_
        prob = model.predict_proba(X_ev)[:, 1]
        m = _metrics(y_ev, prob, f"Fold {i+1}")
        fold_results.append({**m, "best_iter": best_iter})
        print(f"    Best iteration: {best_iter}")

    if not fold_results:
        return {"avg_pr_auc": 0, "avg_best_iter": 300, "folds": []}

    avg_pr   = float(np.mean([r["pr_auc"]    for r in fold_results]))
    avg_iter = int(np.mean([r["best_iter"]   for r in fold_results]))
    print(f"\n  CV summary  avg PR-AUC={avg_pr:.4f}  avg best_iter={avg_iter}")

    result = {"folds": fold_results, "avg_pr_auc": avg_pr, "avg_best_iter": avg_iter}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "cv_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def train_lgbm_final(features: pd.DataFrame, n_estimators: int = 500) -> lgb.LGBMClassifier:
    print(f"\n── LightGBM Final Model (fixed n_estimators={n_estimators}, NO early-stop on val) ──")
    train = features[features["split"] == "train"]
    val   = features[features["split"] == "val"]

    X_tr, y_tr   = _xy(train)
    X_val, y_val = _xy(val)

    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    print(f"  Train: {len(X_tr):,} rows  {int(y_tr.sum())} positive  scale_pos_weight={spw:.1f}")
    print(f"  Val:   {len(X_val):,} rows  {int(y_val.sum())} positive")
    print(f"  Features: {len(FEATURE_COLS)}  (nulls handled natively by LightGBM)")
    print()

    # Train a FIXED number of trees set from the CV best-iteration. We deliberately do NOT
    # early-stop on val: doing so uses val to pick the model, which makes the val metric
    # optimistic and inflates the eventual val→test gap. Val is now a clean held-out set,
    # only read AFTER fitting to report metrics.
    model = lgb.LGBMClassifier(**{**LGBM_BASE, "n_estimators": n_estimators, "scale_pos_weight": spw})
    model.fit(X_tr, y_tr)

    prob_val = model.predict_proba(X_val)[:, 1]
    m = _metrics(y_val, prob_val, "LightGBM final (val 2023-2024)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "lgbm_final.txt"))
    # Persist the exact feature ordering so scoring/explain stay in lock-step with training.
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    imp = pd.DataFrame({
        "feature":          FEATURE_COLS,
        "importance_gain":  model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp.to_parquet(MODELS_DIR / "feature_importance.parquet", index=False)

    print(f"\n  Top 10 features by gain:")
    for _, row in imp.head(10).iterrows():
        bar = "█" * int(row["importance_gain"] / imp["importance_gain"].max() * 20)
        print(f"    {row['feature']:35s}  {bar}")

    with open(MODELS_DIR / "val_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"\n  Saved → {MODELS_DIR}/lgbm_final.txt")
    return model


def run_training(features: pd.DataFrame, skip_cv: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DRUG SHORTAGE RADAR — Model Training")
    print(f"Features: {len(features):,} rows, {len(FEATURE_COLS)} features")
    for split in ["train", "val", "test"]:
        sp  = features[features["split"] == split]
        pos = int(sp[LABEL_COL].sum())
        pct = 100 * pos / max(len(sp), 1)
        print(f"  {split:8s}: {len(sp):>8,} rows  {pos:>5,} positive ({pct:.2f}%)")
    print("=" * 60)

    t0 = time.time()

    lr_metrics = train_logistic(features)

    cv_path = MODELS_DIR / "cv_results.json"
    if skip_cv and cv_path.exists():
        with open(cv_path) as f:
            cv_out = json.load(f)
        print(f"\n── Using cached CV (avg best_iter={cv_out['avg_best_iter']}) ──")
    else:
        cv_out = train_lgbm_cv(features)

    # Fixed tree count = CV best-iteration (no val early-stopping headroom needed anymore).
    avg_iter = cv_out.get("avg_best_iter", 100)
    n_est = max(30, min(400, avg_iter))

    final_model = train_lgbm_final(features, n_estimators=n_est)

    with open(MODELS_DIR / "val_metrics.json") as f:
        lgbm_val = json.load(f)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s")
    print(f"  Baseline logistic   PR-AUC = {lr_metrics['pr_auc']:.4f}")
    print(f"  LightGBM (CV avg)   PR-AUC = {cv_out.get('avg_pr_auc', 0):.4f}")
    print(f"  LightGBM (full val) PR-AUC = {lgbm_val['pr_auc']:.4f}")
    print(f"{'='*60}")
    print(f"\n  Test set is LOCKED. Run scripts/eval_test_LOCKED.py when ready.")

    return {"logistic": lr_metrics, "lgbm_cv": cv_out, "lgbm_val": lgbm_val}


def load_final_model() -> lgb.Booster:
    """Load the trained LightGBM booster from disk."""
    path = MODELS_DIR / "lgbm_final.txt"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run scripts/train.py first.")
    return lgb.Booster(model_file=str(path))


def score_features(features: pd.DataFrame, booster: lgb.Booster = None) -> np.ndarray:
    """Return LightGBM risk scores for every row of `features` (nulls handled natively)."""
    booster = booster or load_final_model()
    X = features[FEATURE_COLS]
    return booster.predict(X)


def save_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """
    Score train+val rows (NEVER test — that split stays locked) and persist a tidy
    predictions table the dashboard and evaluator both consume.

    `risk_score` is an isotonic-calibrated probability so the dashboard's thresholds
    ("risk > 0.3") are meaningful; `risk_score_raw` keeps the uncalibrated margin.
    Calibration is fit IN-SAMPLE on val — it is monotonic, so it changes none of the
    ranking metrics (PR-AUC / ROC / precision@K); its honest assessment is the Brier
    score at locked-test time. Columns: drug_key, month, split, y_6m,
    risk_score, risk_score_raw, next_onset_month.
    """
    from sklearn.isotonic import IsotonicRegression

    booster = load_final_model()
    scored = features[features["split"].isin(["train", "val"])].copy()
    scored["risk_score_raw"] = score_features(scored, booster)

    val = scored[scored["split"] == "val"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val["risk_score_raw"].values, val["y_6m"].astype(int).values)
    scored["risk_score"] = iso.predict(scored["risk_score_raw"].values)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)

    keep = ["drug_key", "month", "split", "y_6m", "risk_score", "risk_score_raw"]
    if "next_onset_month" in scored.columns:
        keep.append("next_onset_month")
    preds = scored[keep].reset_index(drop=True)

    out = DATA_PROCESSED / "predictions_val.parquet"
    preds.to_parquet(out, index=False)
    print(f"  Predictions saved → {out}  ({len(preds):,} rows, "
          f"{int((preds['split'] == 'val').sum()):,} val)  [isotonic-calibrated]")
    return preds


def save_test_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """
    Score the TEST split with the frozen model + saved calibrator and persist
    `predictions_test.parquet`. This is for DISPLAY of the already-spent one-shot test
    result (dashboard figures) — it does NOT retrain, tune, or otherwise touch model
    selection. Same columns as predictions_val.parquet.
    """
    booster = load_final_model()
    with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
        iso = pickle.load(f)

    scored = features[features["split"] == "test"].copy()
    scored["risk_score_raw"] = score_features(scored, booster)
    scored["risk_score"] = iso.predict(scored["risk_score_raw"].values)

    keep = ["drug_key", "month", "split", "y_6m", "risk_score", "risk_score_raw"]
    if "next_onset_month" in scored.columns:
        keep.append("next_onset_month")
    preds = scored[keep].reset_index(drop=True)

    out = DATA_PROCESSED / "predictions_test.parquet"
    preds.to_parquet(out, index=False)
    print(f"  TEST predictions saved → {out}  ({len(preds):,} rows)  "
          f"[frozen one-shot result — display only, NOT for tuning]")
    return preds
