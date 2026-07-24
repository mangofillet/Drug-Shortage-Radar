"""
Plotly figure-builders for the dashboard + static export.

Every function is pure (takes data, returns a styled go.Figure) so they are unit-testable
without a browser and reusable by both app.py and scripts/build_figures.py. A shared light
"report" template keeps everything visually consistent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score,
    brier_score_loss,
)

from config import DATA_PROCESSED
from src.evaluate import precision_at_k, lead_time_analysis

# ── Palette / template ──────────────────────────────────────────────────────
# Editorial "annual-report" monochrome (see Themes/palette.png): off-white canvas,
# ink black, stone greys, one disciplined warm accent reserved for emphasis.
INK = "#3a352e"           # dark warm grey (not pure black) — headline series, ink
AXIS = "#5c564e"          # axis lines / ticks
PAPER = "#ededed"         # signature off-white canvas
CARD = "#ffffff"
GRAY = "#9a9089"          # warm stone grey — secondary series
GRAY_LT = "#c2bbb0"
GRID = "#e6e1d8"
ACCENT = "#bf4e2c"        # terracotta — emphasis / "this model" / positive only

COLOR_VAL = GRAY          # validation series (de-emphasised)
COLOR_TEST = INK          # test series (the headline)
COLOR_PRIMARY = INK
COLOR_ACCENT = ACCENT
COLOR_WARN = ACCENT
COLOR_MUTED = GRAY_LT
BLUES = [INK, "#3c3a37", GRAY, GRAY_LT, "#cfc8bd", "#e0dbd2"]
CAT = [INK, GRAY, ACCENT, GRAY_LT, "#3c3a37", "#cfc8bd"]
FONT = "Inter, 'Helvetica Neue', Arial, -apple-system, 'Segoe UI', sans-serif"
SERIF = "Spectral, Georgia, 'Times New Roman', serif"


def _style(fig: go.Figure, title: str = None, height: int = 380, **kw) -> go.Figure:
    layout = dict(
        template="plotly_white",
        title=dict(text=title, x=0.01, xanchor="left",
                   font=dict(size=17, color=INK, family=SERIF)) if title else None,
        font=dict(family=FONT, size=13, color="#3a3631"),
        margin=dict(l=64, r=24, t=58 if title else 24, b=48),
        height=height,
        plot_bgcolor=CARD,
        paper_bgcolor=CARD,
        colorway=CAT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    layout.update(kw)  # caller kwargs (margin, xaxis, yaxis, barmode, …) override defaults
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS, linewidth=1.1,
                     ticks="outside", tickcolor=GRAY_LT, tickfont=dict(color=AXIS))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=AXIS))
    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=COLOR_MUTED, size=14))
    return _style(fig)


# ── Data loading (used by app + export script) ─────────────────────────────────
def shrink(df):
    """Downcast in place to cut memory ~3x (value-preserving): float64->float32,
    ints downcast, low-cardinality object cols -> category. Skips month/date-like
    object columns so lexical max() on 'YYYY-MM' strings stays chronological."""
    if df is None or getattr(df, "empty", True):
        return df
    n = len(df)
    for c in df.columns:
        col = df[c]; k = col.dtype.kind
        if k == "f":
            df[c] = col.astype("float32")
        elif k in ("i", "u"):
            df[c] = pd.to_numeric(col, downcast="integer")
        elif k == "O":
            if any(t in str(c).lower() for t in ("month", "date", "time", "day")):
                continue
            try:
                if col.nunique(dropna=False) <= n // 2:
                    df[c] = col.astype("category")
            except TypeError:
                pass
    return df


def load_all() -> dict:
    d = {}

    def _rd(name):
        p = DATA_PROCESSED / name
        return shrink(pd.read_parquet(p)) if p.exists() else pd.DataFrame()

    d["shortages"] = _rd("shortages_raw.parquet")
    d["panel"] = _rd("panel.parquet")
    d["val"] = _rd("predictions_val.parquet")
    d["test"] = _rd("predictions_test.parquet")
    d["features"] = _rd("features_with_labels.parquet")
    d["importance"] = _rd("models/feature_importance.parquet")
    if not d["val"].empty:
        d["val"] = d["val"][d["val"]["split"] == "val"]
    return d


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
def fig_shortage_trend(shortages: pd.DataFrame) -> go.Figure:
    if shortages.empty:
        return _empty("No shortage data")
    s = shortages.copy()
    s["year"] = pd.to_datetime(s["onset_date"], errors="coerce").dt.year
    by_year = s.dropna(subset=["year"])
    by_year = by_year[(by_year["year"] >= 2012) & (by_year["year"] <= 2026)]
    counts = by_year.groupby(by_year["year"].astype(int)).size()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=COLOR_PRIMARY,
                           hovertemplate="%{x}: %{y} onsets<extra></extra>"))
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor=COLOR_WARN, opacity=0.08, line_width=0,
                  annotation_text="COVID-19", annotation_position="top left",
                  annotation_font_color=COLOR_WARN)
    return _style(fig, "Drug-shortage onsets per year", height=360,
                  xaxis_title="Onset year", yaxis_title="New shortages")


def fig_shortage_reasons(shortages: pd.DataFrame) -> go.Figure:
    if shortages.empty or "shortage_reason" not in shortages.columns:
        return _empty("No reason data")
    r = shortages["shortage_reason"].replace("", np.nan).dropna()
    r = r[r.str.strip() != ""].value_counts().head(8).sort_values()
    fig = go.Figure(go.Bar(x=r.values, y=r.index, orientation="h", marker_color=COLOR_PRIMARY,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    return _style(fig, "Reported shortage reasons", height=380,
                  xaxis_title="Records", margin=dict(l=300, r=24, t=54, b=48))


def fig_therapeutic_categories(shortages: pd.DataFrame) -> go.Figure:
    if shortages.empty or "therapeutic_category" not in shortages.columns:
        return _empty("No category data")
    c = shortages["therapeutic_category"].replace("", np.nan).dropna()
    c = c[c.str.strip() != ""].value_counts().head(10).sort_values()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=COLOR_ACCENT,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    return _style(fig, "Most-affected therapeutic categories", height=400,
                  xaxis_title="Shortage records", margin=dict(l=200, r=24, t=54, b=48))


def fig_status_donut(shortages: pd.DataFrame) -> go.Figure:
    if shortages.empty or "status_last" not in shortages.columns:
        return _empty("No status data")
    st = shortages["status_last"].fillna("Unknown").str.title().str.strip()
    counts = st.value_counts()
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                           marker=dict(colors=CAT), textinfo="label+percent"))
    return _style(fig, "Shortage status (latest seen)", height=360)


def fig_duration_hist(shortages: pd.DataFrame) -> go.Figure:
    if shortages.empty:
        return _empty("No duration data")
    onset = pd.to_datetime(shortages["onset_date"], errors="coerce")
    resolved = pd.to_datetime(shortages.get("resolved_date_approx"), errors="coerce")
    dur = ((resolved - onset).dt.days / 30.44).dropna()
    dur = dur[(dur >= 0) & (dur <= 120)]
    if dur.empty:
        return _empty("No resolved-shortage durations")
    med = dur.median()
    fig = go.Figure(go.Histogram(x=dur, nbinsx=30, marker_color=COLOR_PRIMARY,
                                 hovertemplate="%{x:.0f} mo: %{y}<extra></extra>"))
    fig.add_vline(x=med, line_dash="dash", line_color=COLOR_WARN,
                  annotation_text=f"median {med:.0f} mo", annotation_position="top")
    return _style(fig, "How long resolved shortages lasted", height=360,
                  xaxis_title="Duration (months)", yaxis_title="Shortages")


def fig_base_rate_over_time(panel: pd.DataFrame) -> go.Figure:
    if panel.empty:
        return _empty("No panel data")
    p = panel.copy()
    p["q"] = pd.to_datetime(p["month"]).dt.to_period("Q")
    br = p.groupby("q")["y_6m"].mean() * 100
    x = br.index.to_timestamp()
    fig = go.Figure(go.Scatter(x=x, y=br.values, mode="lines+markers",
                               line=dict(color=INK, width=2),
                               marker=dict(size=5),
                               hovertemplate="%{x|%Y-Q%q}: %{y:.2f}%<extra></extra>"))
    # shade splits: val = stone grey, test = accent tint
    fig.add_vrect(x0="2023-01-01", x1="2025-01-01", fillcolor=GRAY, opacity=0.13,
                  line_width=0, annotation_text="VAL", annotation_position="top left")
    fig.add_vrect(x0="2025-01-01", x1=str(x.max().date()), fillcolor=ACCENT, opacity=0.13,
                  line_width=0, annotation_text="TEST", annotation_position="top left")
    return _style(fig, "Label base rate over time (non-stationarity)", height=340,
                  xaxis_title="Quarter", yaxis_title="% drug-months going short (6 mo)")


def fig_top_shorted_drugs(shortages: pd.DataFrame, n: int = 15) -> go.Figure:
    if shortages.empty:
        return _empty("No shortage data")
    s = shortages.copy()
    s = s[pd.to_datetime(s["onset_date"], errors="coerce").notna()]
    key = s.get("generic_name_norm", s["generic_name"]).fillna(s["generic_name"])
    top = key.str.title().value_counts().head(n).sort_values()
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=COLOR_PRIMARY,
                           hovertemplate="%{y}: %{x} onsets<extra></extra>"))
    return _style(fig, f"Most chronically-shorted drugs (top {n})", height=460,
                  xaxis_title="Distinct shortage onsets", margin=dict(l=240, r=24, t=54, b=48))


# ══════════════════════════════════════════════════════════════════════════════
# Model performance (val vs test)
# ══════════════════════════════════════════════════════════════════════════════
def _yscore(df: pd.DataFrame):
    """y, ranking score (raw)."""
    return df["y_6m"].astype(int).values, df["risk_score_raw"].values


def fig_pr_curve(val: pd.DataFrame, test: pd.DataFrame, k_op: int = 50) -> go.Figure:
    fig = go.Figure()
    for df, name, col in [(val, "Val 2023–24", COLOR_VAL), (test, "Test 2025", COLOR_TEST)]:
        if df is None or df.empty:
            continue
        y, s = _yscore(df)
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP={ap:.3f})",
                                 line=dict(color=col, width=2.5, dash="dash" if name.startswith("Val") else "solid")))
        fig.add_hline(y=y.mean(), line_dash="dot", line_color=col, opacity=0.4)
        # Operating point: where the top-K watchlist sits on the curve (inline label, no arrow)
        order = np.argsort(s)[::-1][:k_op]
        tp = int(y[order].sum())
        p_op, r_op = tp / k_op, tp / max(y.sum(), 1)
        fig.add_trace(go.Scatter(x=[r_op], y=[p_op], mode="markers+text", showlegend=False,
                                 text=[f"  {name.split()[0]} {p_op*100:.0f}%"], textposition="middle right",
                                 textfont=dict(color=ACCENT, size=12, family=FONT),
                                 marker=dict(color=ACCENT, size=12, symbol="star", line=dict(color="white", width=1.5)),
                                 hovertemplate=f"{name} top-{k_op}: precision %{{y:.2f}}, recall %{{x:.3f}}<extra></extra>"))
    fig.add_annotation(x=0.98, y=0.96, xref="paper", yref="paper", showarrow=False, xanchor="right",
                       text=f"★ top-{k_op} watchlist 'ship point'", font=dict(size=11, color=ACCENT))
    return _style(fig, "Precision–Recall curve", height=380,
                  xaxis_title="Recall", yaxis_title="Precision")


def fig_roc_curve(val: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for df, name, col in [(val, "Val 2023–24", COLOR_VAL), (test, "Test 2025", COLOR_TEST)]:
        if df is None or df.empty:
            continue
        y, s = _yscore(df)
        fpr, tpr, _ = roc_curve(y, s)
        auc_v = roc_auc_score(y, s)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_v:.3f})",
                                 line=dict(color=col, width=2.5, dash="dash" if name.startswith("Val") else "solid")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line=dict(color=COLOR_MUTED, dash="dash", width=1)))
    return _style(fig, "ROC curve", height=380, xaxis_title="False-positive rate",
                  yaxis_title="True-positive rate")


def fig_calibration(val: pd.DataFrame, test: pd.DataFrame, bins: int = 10) -> go.Figure:
    fig = go.Figure()
    for df, name, col in [(val, "Val 2023–24", COLOR_VAL), (test, "Test 2025", COLOR_TEST)]:
        if df is None or df.empty:
            continue
        y = df["y_6m"].astype(int).values
        p = df["risk_score"].values  # calibrated
        # quantile bins (scores are tiny + skewed)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        rows = pd.DataFrame({"b": idx, "p": p, "y": y}).groupby("b").agg(mp=("p", "mean"), my=("y", "mean"))
        brier = brier_score_loss(y, p)
        fig.add_trace(go.Scatter(x=rows["mp"], y=rows["my"], mode="lines+markers",
                                 name=f"{name} (Brier={brier:.3f})", line=dict(color=col, width=2.5, dash="dash" if name.startswith("Val") else "solid")))
    lim = max(0.01, *(t.y.max() if hasattr(t, "y") and t.y is not None and len(t.y) else 0 for t in fig.data)) if fig.data else 0.1
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect",
                             line=dict(color=COLOR_MUTED, dash="dash", width=1)))
    return _style(fig, "Calibration (reliability)", height=380,
                  xaxis_title="Mean predicted risk", yaxis_title="Observed shortage rate")


def fig_precision_at_k(val: pd.DataFrame, test: pd.DataFrame, k_op: int = 50) -> go.Figure:
    ks = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    fig = go.Figure()
    for df, name, col in [(val, "Val 2023–24", COLOR_VAL), (test, "Test 2025", COLOR_TEST)]:
        if df is None or df.empty:
            continue
        y, s = _yscore(df)
        ys = [precision_at_k(y, s, k=k) * 100 for k in ks]
        fig.add_trace(go.Scatter(x=ks, y=ys, mode="lines+markers", name=name,
                                 line=dict(color=col, width=2.5, dash="dash" if name.startswith("Val") else "solid")))
        fig.add_hline(y=y.mean() * 100, line_dash="dot", line_color=col, opacity=0.4)
        # Operating point — the watchlist size the tool actually ships (inline label, no arrow)
        p_op = precision_at_k(y, s, k=k_op) * 100
        tpos = "bottom center" if name.startswith("Test") else "top center"
        fig.add_trace(go.Scatter(x=[k_op], y=[p_op], mode="markers+text", showlegend=False,
                                 text=[f"{p_op:.0f}%"], textposition=tpos,
                                 textfont=dict(color=ACCENT, size=12, family=FONT),
                                 marker=dict(color=ACCENT, size=12, symbol="star",
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate=f"{name}: precision@{k_op}=%{{y:.0f}}%<extra></extra>"))
    fig.add_vline(x=k_op, line_dash="dot", line_color=ACCENT, opacity=0.5)
    fig.add_annotation(x=0.98, y=0.96, xref="paper", yref="paper", showarrow=False, xanchor="right",
                       text=f"★ top-{k_op} 'ship point' · dotted = base rate", font=dict(size=11, color=ACCENT))
    return _style(fig, "Watchlist precision@K (deployment metric)", height=380,
                  xaxis_title="Watchlist size K", yaxis_title="Precision (%)")


def fig_lead_time_hist(val: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for df, name, col in [(val, "Val 2023–24", COLOR_VAL), (test, "Test 2025", COLOR_TEST)]:
        if df is None or df.empty or "next_onset_month" not in df.columns:
            continue
        d = df.copy()
        d["risk_score"] = d["risk_score_raw"]
        lead = lead_time_analysis(d, k=50)
        hist = lead.get("lead_histogram", {})
        if not hist:
            continue
        xs = list(range(1, 7))
        ys = [hist.get(m, 0) for m in xs]
        fig.add_trace(go.Bar(x=xs, y=ys, name=f"{name} (μ={lead.get('mean_lead_months')} mo)",
                             marker_color=col, opacity=0.85))
    fig.update_layout(barmode="group")
    return _style(fig, "Advance warning: months before onset (top-50 watchlist)", height=360,
                  xaxis_title="Months before onset first flagged", yaxis_title="Onsets caught")


def fig_feature_importance(importance: pd.DataFrame, n: int = 15) -> go.Figure:
    if importance is None or importance.empty:
        return _empty("No importance data")
    imp = importance.sort_values("importance_gain", ascending=True).tail(n)
    fig = go.Figure(go.Bar(x=imp["importance_gain"], y=imp["feature"], orientation="h",
                           marker_color=COLOR_PRIMARY, hovertemplate="%{y}: %{x:.0f}<extra></extra>"))
    return _style(fig, "Feature importance (LightGBM gain)", height=460,
                  xaxis_title="Total gain", margin=dict(l=220, r=24, t=54, b=48))


def fig_shap_summary(features: pd.DataFrame, sample: int = 1500) -> go.Figure:
    if features is None or features.empty:
        return _empty("No feature data")
    try:
        from src.explain import compute_shap
        val = features[features["split"] == "val"]
        if len(val) > sample:
            val = val.sample(sample, random_state=42)
        shap_df = compute_shap(val)
        mean_abs = (shap_df.groupby("feature")["shap_value"].apply(lambda s: s.abs().mean())
                    .sort_values(ascending=True).tail(15))
    except Exception as e:  # noqa: BLE001
        return _empty(f"SHAP unavailable: {e}")
    fig = go.Figure(go.Bar(x=mean_abs.values, y=mean_abs.index, orientation="h",
                           marker_color=COLOR_ACCENT, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    return _style(fig, "Mean |SHAP| — drivers of predicted risk", height=460,
                  xaxis_title="Mean |SHAP value|", margin=dict(l=220, r=24, t=54, b=48))


def fig_val_test_metrics(val: pd.DataFrame, test: pd.DataFrame) -> go.Figure:
    def _m(df):
        y, s = _yscore(df)
        return {
            "PR-AUC": average_precision_score(y, s),
            "ROC-AUC": roc_auc_score(y, s),
            "Precision@50": precision_at_k(y, s, k=50),
            "Brier (cal.)": brier_score_loss(y, df["risk_score"].values),
        }
    if val is None or val.empty or test is None or test.empty:
        return _empty("Need val + test predictions")
    mv, mt = _m(val), _m(test)
    labels = list(mv.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[mv[k] for k in labels], name="Val 2023–24",
                         marker_color=COLOR_VAL, text=[f"{mv[k]:.2f}" for k in labels], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[mt[k] for k in labels], name="Test 2025",
                         marker_color=COLOR_TEST, text=[f"{mt[k]:.2f}" for k in labels], textposition="outside"))
    fig.update_layout(barmode="group")
    return _style(fig, "Validation vs. locked test", height=380, yaxis_title="Score")


def fig_score_distribution(val: pd.DataFrame) -> go.Figure:
    if val is None or val.empty:
        return _empty("No predictions")
    fig = go.Figure()
    for flag, name, col in [(False, "No shortage", COLOR_MUTED), (True, "Went short", COLOR_WARN)]:
        d = val[val["y_6m"] == flag]["risk_score"]
        fig.add_trace(go.Histogram(x=d, name=name, marker_color=col, opacity=0.7, nbinsx=40,
                                   histnorm="probability density"))
    fig.update_layout(barmode="overlay")
    return _style(fig, "Calibrated risk score by outcome (val)", height=340,
                  xaxis_title="Calibrated risk score", yaxis_title="Density")


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark vs prior work
# ══════════════════════════════════════════════════════════════════════════════
# (study, horizon_weeks, AUROC or None, data_type, reported_metric, reported_value)
# Only clearly-attributable published results; AUROC populated where the study reports it.
_STUDIES = [
    ("UC Berkeley 2024", 4, 0.93, "Public FDA", "AUROC", 0.93),
    ("Canadian XGBoost 2023", 4.3, None, "Proprietary sales", "Cohen's κ", 0.44),
    ("South Korea 2025", None, None, "Regulatory", "F1", 0.70),
    ("This model (TEST)", 26, 0.83, "Public FDA+Wayback", "ROC-AUC", 0.83),
]


def fig_benchmark_positioning() -> go.Figure:
    fig = go.Figure()
    for name, wk, auroc, dtype, *_ in _STUDIES:
        if wk is None or auroc is None:
            continue
        is_this = name.startswith("This model")
        fig.add_trace(go.Scatter(
            x=[wk], y=[auroc], mode="markers+text", name=name, text=["  " + name],
            textposition="middle right",
            marker=dict(size=26 if is_this else 14,
                        color=ACCENT if is_this else INK,
                        line=dict(width=2, color="white"), symbol="star" if is_this else "circle"),
            showlegend=False,
            hovertemplate=f"{name}<br>horizon %{{x}} wk<br>AUROC %{{y}}<extra></extra>"))
    fig.update_xaxes(type="log", title="Forecast horizon (weeks, log scale)", range=[np.log10(3), np.log10(60)])
    fig.add_annotation(x=np.log10(26), y=0.83, ax=0, ay=-46, text="hardest combo:<br>public data, 6-mo horizon",
                       arrowcolor=COLOR_TEST, font=dict(color=COLOR_TEST, size=12), align="center")
    return _style(fig, "Positioning: forecast horizon vs. AUROC", height=400,
                  yaxis_title="ROC-AUC", yaxis=dict(range=[0.7, 1.0]))


def fig_benchmark_metrics() -> go.Figure:
    rows = [(n, val, m, dt) for (n, wk, au, dt, m, val) in _STUDIES]
    rows = sorted(rows, key=lambda r: r[1])
    colors = [ACCENT if n.startswith("This model") else INK for n, *_ in rows]
    labels = [f"{n}" for n, *_ in rows]
    text = [f"{m} {v:.2f}" for (n, v, m, dt) in rows]
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=labels, orientation="h",
                           marker_color=colors, text=text, textposition="outside",
                           hovertemplate="%{y}<br>%{text}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.05])
    return _style(fig, "Reported headline metric by study (⚠ metrics differ)", height=360,
                  xaxis_title="Reported value (AUROC / κ / F1 — not identical metrics)",
                  margin=dict(l=200, r=60, t=54, b=60))


# Registry for the export script (name → builder needing only loaded data)
def build_all(data: dict) -> dict:
    s, p, v, t = data["shortages"], data["panel"], data["val"], data["test"]
    imp, feats = data["importance"], data["features"]
    return {
        "eda_shortage_trend": fig_shortage_trend(s),
        "eda_reasons": fig_shortage_reasons(s),
        "eda_categories": fig_therapeutic_categories(s),
        "eda_status": fig_status_donut(s),
        "eda_duration": fig_duration_hist(s),
        "eda_base_rate": fig_base_rate_over_time(p),
        "eda_top_drugs": fig_top_shorted_drugs(s),
        "perf_pr_curve": fig_pr_curve(v, t),
        "perf_roc_curve": fig_roc_curve(v, t),
        "perf_calibration": fig_calibration(v, t),
        "perf_precision_at_k": fig_precision_at_k(v, t),
        "perf_lead_time": fig_lead_time_hist(v, t),
        "perf_importance": fig_feature_importance(imp),
        "perf_shap": fig_shap_summary(feats),
        "perf_val_test": fig_val_test_metrics(v, t),
        "perf_score_dist": fig_score_distribution(v),
        "bench_positioning": fig_benchmark_positioning(),
        "bench_metrics": fig_benchmark_metrics(),
    }
