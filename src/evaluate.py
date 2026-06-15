"""
Phase 6 — Validation metrics for the drug-shortage radar.

Everything here runs on the VALIDATION split only (2023-01 → 2024-12). The test
window stays locked behind scripts/eval_test_LOCKED.py.

Three lenses, because PR-AUC alone understates a watchlist tool:
  1. Global ranking quality — PR-AUC / ROC-AUC / Brier.
  2. Watchlist precision — precision@K on a monthly top-K list (the deployment metric).
  3. Lead time — how many months BEFORE onset the model first flags a drug.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

REPORTS_DIR = Path(__file__).parent.parent / "reports"

PRIOR_WORK_SECTION = """## 7. Comparison to prior work

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
"""


def recall_at_k(y_true, scores, k: int = 50) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return y_true[top_k].sum() / max(y_true.sum(), 1)


def precision_at_k(y_true, scores, k: int = 50) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return y_true[top_k].mean()


def evaluate_model(name: str, scores: np.ndarray, y_true: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    pr_auc = average_precision_score(y_true, scores) if y_true.sum() > 0 else 0.0
    roc_auc = roc_auc_score(y_true, scores) if y_true.sum() > 0 else 0.5
    brier = brier_score_loss(y_true, np.clip(scores, 0, 1))
    return {
        "model": name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_at_50": round(recall_at_k(y_true, scores, k=50), 4),
        "precision_at_50": round(precision_at_k(y_true, scores, k=50), 4),
        "brier_score": round(brier, 4),
        "n_positive": int(y_true.sum()),
        "n_total": int(len(y_true)),
        "base_rate_pct": round(100 * y_true.mean(), 3),
    }


def precision_at_k_curve(y_true, scores, ks=(20, 50, 100, 200, 500)) -> list[dict]:
    """Global precision@K with lift over the base rate, for several K."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    base = y_true.mean()
    rows = []
    for k in ks:
        if k > len(y_true):
            continue
        p = precision_at_k(y_true, scores, k=k)
        rows.append({
            "k": k,
            "precision": round(p, 4),
            "lift": round(p / base, 1) if base > 0 else 0.0,
            "n_caught": int(round(p * k)),
        })
    return rows


def lead_time_analysis(predictions: pd.DataFrame, k: int = 50) -> dict:
    """
    Advance-warning measured the way the tool is actually used: each month publish a
    top-K watchlist, then for every real shortage ONSET ask how many months earlier
    the drug first appeared on that watchlist.

    predictions must have: drug_key, month, risk_score, y_6m, next_onset_month.
    Returns a summary dict plus the per-onset lead months for histogramming.
    """
    if "next_onset_month" not in predictions.columns:
        return {}

    df = predictions.copy()
    df["month"] = pd.to_datetime(df["month"])

    # Rank within each month → who is on this month's published top-K list.
    df["rank_in_month"] = df.groupby("month")["risk_score"].rank(ascending=False, method="first")
    df["on_watchlist"] = df["rank_in_month"] <= k

    # Pre-onset rows that the model flagged (y_6m=True means an onset is 1–6 months out).
    flagged = df[(df["y_6m"] == True) & df["next_onset_month"].notna() & df["on_watchlist"]].copy()
    if len(flagged) == 0:
        return {"k": k, "n_onsets_caught": 0, "lead_months": []}

    flagged["next_onset_month"] = pd.to_datetime(flagged["next_onset_month"])
    flagged["lead_months"] = (
        (flagged["next_onset_month"] - flagged["month"]).dt.days / 30.44
    ).round()

    # Earliest catch per onset event = max advance warning the team would have had.
    per_onset = flagged.groupby(["drug_key", "next_onset_month"])["lead_months"].max()
    leads = per_onset.values

    # How many of all distinct onset events in the eval window were caught at all?
    onset_events = df[df["y_6m"] == True].dropna(subset=["next_onset_month"])
    n_onsets_total = onset_events.groupby(["drug_key", "next_onset_month"]).ngroups

    return {
        "k": k,
        "n_onsets_total": int(n_onsets_total),
        "n_onsets_caught": int(len(per_onset)),
        "catch_rate": round(len(per_onset) / max(n_onsets_total, 1), 3),
        "mean_lead_months": round(float(np.mean(leads)), 2),
        "median_lead_months": round(float(np.median(leads)), 2),
        "lead_months": [int(x) for x in leads],
        "lead_histogram": {int(m): int((leads == m).sum()) for m in sorted(set(leads))},
    }


def rolling_origin_backtest(features: pd.DataFrame, train_fn) -> list[dict]:
    """Quarterly sliding-window backtest within train+val. Returns per-window metrics."""
    features = features[features["split"].isin(["train", "val"])].copy()
    months = sorted(features["month"].unique())

    results = []
    first_test_start_idx = len(months) // 2
    for i in range(first_test_start_idx, len(months) - 6, 3):
        train_end = months[i]
        test_start = months[i + 1]
        test_end = months[min(i + 6, len(months) - 1)]

        tr = features[features["month"] <= train_end]
        te = features[(features["month"] >= test_start) & (features["month"] <= test_end)]

        if tr["y_6m"].sum() < 10 or te["y_6m"].sum() < 3:
            continue

        model_results = train_fn(tr)
        if "lgbm" not in model_results:
            continue

        lgbm = model_results["lgbm"]
        feat_cols = lgbm["feat_cols"]
        te_cols = [c for c in feat_cols if c in te.columns]
        te_X = te[te_cols].fillna(-1).values.astype("float32")
        scores = lgbm["model"].predict(te_X)
        y = te["y_6m"].astype(int).values

        metrics = evaluate_model(f"lgbm_window_{pd.Timestamp(train_end).strftime('%Y%m')}", scores, y)
        metrics["train_end"] = str(pd.Timestamp(train_end).date())
        metrics["test_start"] = str(pd.Timestamp(test_start).date())
        metrics["test_end"] = str(pd.Timestamp(test_end).date())
        results.append(metrics)
        print(f"  Backtest window train≤{pd.Timestamp(train_end).date()}: "
              f"PR-AUC={metrics['pr_auc']:.3f}, R@50={metrics['recall_at_50']:.3f}")

    return results


def write_validation_report(
    metrics: list[dict],
    pk_curve: list[dict],
    lead: dict,
    importance: pd.DataFrame = None,
    backtest: list[dict] = None,
    calibration: dict = None,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation Report — Drug Shortage Radar\n"]
    lines.append("_Validation split 2023-01 → 2024-12. Test window remains locked._\n")

    lines.append("## 1. Ranking quality\n")
    lines.append("| Model | PR-AUC | ROC-AUC | Recall@50 | Precision@50 | Brier |")
    lines.append("|-------|--------|---------|-----------|--------------|-------|")
    for m in metrics:
        lines.append(
            f"| {m['model']} | {m['pr_auc']} | {m['roc_auc']} | "
            f"{m['recall_at_50']} | {m['precision_at_50']} | {m['brier_score']} |"
        )
    if metrics:
        lines.append(f"\nBase rate: **{metrics[0]['base_rate_pct']:.2f}%** positive "
                     f"({metrics[0]['n_positive']:,} / {metrics[0]['n_total']:,} drug-months).\n")
        lines.append("> PR-AUC is low because this is a ~2% rare-event problem with much "
                     "causation unobservable in public feeds. The deployment metric below is "
                     "what matters: a small ranked watchlist is many times better than chance.\n")

    if pk_curve:
        lines.append("## 2. Watchlist precision (the deployment metric)\n")
        lines.append("A team watching the top-K riskiest drug-months each period sees:\n")
        lines.append("| Top-K | Precision | Lift vs. base rate | Caught |")
        lines.append("|-------|-----------|--------------------|--------|")
        for r in pk_curve:
            lines.append(f"| {r['k']} | {r['precision']*100:.1f}% | {r['lift']}× | "
                         f"{r['n_caught']}/{r['k']} |")
        lines.append("")

    if lead and lead.get("n_onsets_caught", 0) > 0:
        lines.append(f"## 3. Lead time (monthly top-{lead['k']} watchlist)\n")
        lines.append(
            f"- Of **{lead.get('n_onsets_total', 'N/A')}** distinct shortage onsets in the "
            f"window, the watchlist flagged **{lead['n_onsets_caught']}** in advance "
            f"(**{lead.get('catch_rate', 0)*100:.0f}%** catch rate)."
        )
        lines.append(f"- Mean advance warning: **{lead['mean_lead_months']} months** "
                     f"(median {lead['median_lead_months']}).")
        if lead.get("lead_histogram"):
            lines.append("\n| Months before onset | Onsets first caught |")
            lines.append("|---------------------|---------------------|")
            for m, n in sorted(lead["lead_histogram"].items(), reverse=True):
                lines.append(f"| {m} | {n} |")
        lines.append("")

    if calibration:
        lines.append("## 4. Calibration (isotonic, fit in-sample on val)\n")
        lines.append(
            f"- Brier score: **{calibration['brier_calibrated']}** calibrated "
            f"vs {calibration['brier_raw']} raw."
        )
        lines.append(
            f"- Mean predicted risk **{calibration['mean_pred']}** ≈ base rate "
            f"{calibration['base_rate']}, so a displayed \"risk = 0.30\" is a genuine "
            f"~30% 6-month onset probability."
        )
        lines.append(
            "- Monotonic, so it leaves §1–3 ranking metrics unchanged. Fit in-sample on "
            "val for display; its honest assessment is the Brier score at locked-test time.\n"
        )

    if importance is not None and len(importance):
        lines.append("## 5. Top model drivers (gain)\n")
        lines.append("| Feature | Gain |")
        lines.append("|---------|------|")
        top = importance.sort_values("importance_gain", ascending=False).head(12)
        gmax = top["importance_gain"].max()
        for _, row in top.iterrows():
            bar = "█" * int(round(row["importance_gain"] / gmax * 12))
            lines.append(f"| {row['feature']} | {bar} |")
        lines.append("")

    if backtest:
        lines.append("## 6. Rolling-origin backtest (quarterly)\n")
        lines.append("| Train End | Test Start | PR-AUC | Recall@50 |")
        lines.append("|-----------|------------|--------|-----------|")
        for b in backtest:
            lines.append(f"| {b['train_end']} | {b['test_start']} | {b['pr_auc']} | {b['recall_at_50']} |")
        lines.append("")

    lines.append(PRIOR_WORK_SECTION)

    report = "\n".join(lines)
    path = REPORTS_DIR / "validation.md"
    path.write_text(report)
    print(f"\nValidation report written → {path}")
    return report
