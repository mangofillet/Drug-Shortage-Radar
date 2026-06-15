"""
Phase 7 — Plotly Dash dashboard: Watchlist / Drug Detail / Model Card.
Runs fully offline from data/processed/ parquet caches.
Usage: python app.py
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

from src import figures as F

# ── Data loading (module-level, offline from parquet caches) ──────────────────

DATA_DIR = Path(__file__).parent / "data" / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = Path(__file__).parent / "reports"


def _load_data():
    paths = {
        "features": DATA_DIR / "features_with_labels.parquet",
        "shortages": DATA_DIR / "shortages_raw.parquet",
        "shortages_live": DATA_DIR / "shortages_live.parquet",
        "inspections": DATA_DIR / "inspections.parquet",
        "enforcement": DATA_DIR / "enforcement.parquet",
        "compliance": DATA_DIR / "compliance.parquet",
    }
    data = {}
    for key, path in paths.items():
        if path.exists():
            data[key] = pd.read_parquet(path)
        else:
            data[key] = pd.DataFrame()
    return data


def _load_predictions(data: dict) -> pd.DataFrame:
    """Load or generate risk scores."""
    score_path = DATA_DIR / "predictions_val.parquet"
    if score_path.exists():
        return pd.read_parquet(score_path)

    features = data.get("features", pd.DataFrame())
    if features.empty or "risk_score" not in features.columns:
        # Fallback: use past_shortage_count as a proxy score so the UI still renders
        if not features.empty and "past_shortage_count" in features.columns:
            features = features.copy()
            features["risk_score"] = features["past_shortage_count"] / (features["past_shortage_count"].max() + 1)
        else:
            return pd.DataFrame()

    return features[["drug_key", "month", "split", "y_6m", "risk_score"]]


DATA = _load_data()
PREDS = _load_predictions(DATA)
FEAT = DATA.get("features", pd.DataFrame())
SHORTAGES = DATA.get("shortages", pd.DataFrame())
LIVE = DATA.get("shortages_live", pd.DataFrame())


def _latest_predictions(split: str = "val") -> pd.DataFrame:
    """Get the most recent month's predictions for a split."""
    if PREDS.empty:
        return pd.DataFrame()
    sp = PREDS[PREDS["split"] == split]
    if sp.empty:
        return pd.DataFrame()
    latest_month = sp["month"].max()
    return sp[sp["month"] == latest_month].sort_values("risk_score", ascending=False)


def _currently_short() -> set:
    """Drug keys currently in shortage from live feed."""
    if LIVE.empty:
        return set()
    live_current = LIVE[LIVE.get("status", pd.Series()) == "Current"] if "status" in LIVE.columns else LIVE
    return set(live_current["generic_name"].str.lower().str.strip().tolist())


CURRENTLY_SHORT = _currently_short()

# Data for the analytical tabs (parquet reads only; figures built lazily per tab).
FIGDATA = F.load_all()


def _graph(fig, **kw):
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, **kw)


def _kpi_card(value, label, accent=False):
    return dbc.Col(html.Div([
        html.Div(value, className="num"),
        html.Div(label, className="lbl"),
    ], className="kpi accent" if accent else "kpi"), md=True)


def _section(eyebrow, title, lead=None):
    items = [html.Div(eyebrow, className="section-eyebrow"), html.H4(title, className="section")]
    if lead:
        items.append(html.P(lead, className="lead"))
    items.append(html.Hr(className="section-rule"))
    return html.Div(items, style={"marginBottom": "20px"})


def _takeaway(title, body):
    return dbc.Col(html.Div([
        html.Div(title, className="section-eyebrow", style={"marginBottom": "7px"}),
        html.Div(body, style={"fontSize": "13px", "color": "#3a3631", "lineHeight": "1.55"}),
    ], className="kpi accent", style={"textAlign": "left"}), md=4)


def _navtile(title, desc):
    return dbc.Col(html.Div([
        html.Div(title, style={"fontFamily": "Spectral, serif", "fontSize": "18px",
                               "fontWeight": "600", "color": "#141414", "marginBottom": "3px"}),
        html.Div(desc, style={"fontSize": "13.5px", "color": "#5c564e", "lineHeight": "1.5"}),
    ], style={"borderLeft": "2px solid #d8d2c7", "padding": "2px 0 2px 15px", "height": "100%"}), md=4)


_BASE_RATE = 0.022  # ~2.2% average drug-month shortage onset rate


def _risk_band(score: float):
    """Map a calibrated risk to (label, colour, pct, lift-over-baseline)."""
    pct, lift = score * 100, score / _BASE_RATE
    if score < 0.05:
        lab, col = "Low risk", "#2e7d32"
    elif score < 0.15:
        lab, col = "Moderate risk", "#b07d2b"
    elif score < 0.30:
        lab, col = "Elevated risk", "#bf4e2c"
    else:
        lab, col = "High risk", "#9b2d1f"
    return lab, col, pct, lift


def _risk_readout(score: float):
    lab, col, pct, lift = _risk_band(score)
    return html.Div([
        html.Div(lab.upper(), className="section-eyebrow", style={"color": col, "marginBottom": "4px"}),
        html.Div(f"{pct:.0f}%", style={"fontFamily": "Spectral, serif", "fontSize": "72px",
                                       "fontWeight": "700", "color": col, "lineHeight": "1"}),
        html.Div([
            "modeled probability of a ", html.B("new"), " shortage beginning within 6 months",
            html.Span(f"  ·  ≈ {lift:.0f}× the ~2% average drug-month",
                      style={"color": "#7a736b"}),
        ], style={"color": "#3a3631", "fontSize": "14px", "marginTop": "8px", "maxWidth": "560px"}),
    ], className="kpi", style={"textAlign": "left", "borderTopColor": col, "borderTopWidth": "3px",
                               "padding": "20px 24px", "marginBottom": "22px"})


_RISK_EXPLAINER = (
    "**What “risk” means here.** It is the model’s *calibrated probability* that this drug "
    "(this generic + route) begins a **new** shortage within the next six months — not whether "
    "it is short today.\n\n"
    "**How it’s calculated.** A LightGBM model scores every drug-month from public FDA signals: "
    "how many manufacturers still make it (market concentration), recent inspection findings and "
    "recalls (manufacturing quality), price erosion (CMS NADAC), and timing signals — months "
    "since the last recall or shortage, and whether related forms of the same molecule are "
    "currently short. The raw score is then **isotonic-calibrated** on held-out data, so “12%” "
    "genuinely means about 12 in 100.\n\n"
    "**How to use it.** The average drug-month sits near **2%**, so any double-digit figure is "
    "well above baseline. Bands: under 5% low · 5–15% moderate · 15–30% elevated · above 30% high. "
    "Use it to decide where to look first — which drugs to pre-order, qualify a second supplier "
    "for, or raise par stock on — not as a guarantee. It flags *where* to spend attention, not a "
    "certainty that a shortage will happen."
)


def _drug_key_to_label(dk: str, max_len: int = 56) -> str:
    parts = dk.split("|")
    name = parts[0].title().strip()
    # Combination products (e.g. IV multivitamins) carry their full ingredient list as the
    # NDC generic_name; abbreviate for the dropdown/labels so one drug doesn't dominate.
    if len(name) > max_len:
        name = name[:max_len].rsplit(",", 1)[0].rstrip(" ,") + " … (combination)"
    route = parts[1].title() if len(parts) > 1 else ""
    return f"{name} ({route})" if route else name


# ── Layout ────────────────────────────────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True, title="Drug Shortage Radar")

# Dossier / report aesthetic — warm paper, white sheet, serif headings, one ink + one
# terracotta accent. No decorative imagery; figures are the only graphics.
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root{--ink:#141414;--paper:#e7e3da;--accent:#bf4e2c;--gray:#7a736b;--line:#d8d2c7;}
html,body{background:var(--paper);margin:0;font-family:'Inter',system-ui,sans-serif;color:#2b2825;-webkit-font-smoothing:antialiased;}
.dossier{max-width:1180px;margin:30px auto;background:#fff;box-shadow:0 2px 22px rgba(20,18,16,.12);border:1px solid var(--line);}
.masthead{background:var(--ink);color:#f3efe7;padding:34px 44px 28px;border-bottom:3px solid var(--accent);}
.masthead .eyebrow{font-size:12px;letter-spacing:.26em;text-transform:uppercase;color:#b3a99b;margin-bottom:14px;font-weight:500;}
.masthead h1{font-family:'Spectral',serif;font-weight:700;font-size:40px;letter-spacing:.005em;margin:0;color:#fff;line-height:1.05;}
.masthead .sub{font-size:16px;color:#d2cabb;margin-top:11px;letter-spacing:.01em;}
.tabwrap{padding:0 44px;border-bottom:1px solid var(--line);background:#faf8f4;}
.bodywrap{padding:36px 44px 44px;}
.section-eyebrow{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:700;}
h4.section{font-family:'Spectral',serif;font-weight:600;font-size:31px;color:var(--ink);margin:5px 0 8px;line-height:1.12;}
.lead{color:#5c564e;font-size:16.5px;max-width:760px;line-height:1.6;}
hr{border:0;border-top:1px solid var(--line);margin:24px 0;}
.kpi{border:1px solid var(--line);border-top:3px solid var(--ink);padding:20px 14px;text-align:center;background:#fff;height:100%;}
.kpi.accent{border-top-color:var(--accent);}
.kpi .num{font-family:'Spectral',serif;font-weight:700;font-size:42px;line-height:1;color:var(--ink);}
.kpi.accent .num{color:var(--accent);}
.kpi .lbl{font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--gray);margin-top:11px;}
.bodywrap a{color:var(--accent);}
.note{font-size:13.5px;color:#5c564e;border-left:3px solid var(--line);padding:11px 18px;background:#faf8f4;font-family:'Inter',sans-serif;line-height:1.55;}
.note.warn{border-left-color:var(--accent);}
.section-rule{border:0;border-top:1.5px solid var(--ink);margin:0 0 22px;}
/* uniform dossier typography for all rendered markdown (model card, reports, prose) */
.bodywrap h1,.bodywrap h2,.bodywrap h3,.bodywrap h4,.bodywrap h5{font-family:'Spectral',serif;color:var(--ink);font-weight:600;}
.bodywrap h1{font-size:25px;margin:8px 0 10px;}
.bodywrap h2{font-size:21px;margin:32px 0 11px;padding-bottom:7px;border-bottom:1px solid var(--line);}
.bodywrap h3{font-size:17px;margin:20px 0 8px;}
.bodywrap p,.bodywrap li{font-family:'Inter',sans-serif;font-size:15.5px;color:#3a3631;line-height:1.65;}
.bodywrap strong{color:var(--ink);font-weight:600;}
.bodywrap em{color:#5c564e;}
.bodywrap table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px;font-family:'Inter',sans-serif;}
.bodywrap thead th{text-align:left;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:#5c564e;border-bottom:2px solid var(--ink);padding:9px 12px;}
.bodywrap tbody td{padding:8px 12px;border-bottom:1px solid var(--line);color:#3a3631;}
.bodywrap tbody tr:nth-child(even){background:#faf8f4;}
.bodywrap blockquote{border-left:3px solid var(--accent);margin:16px 0;padding:8px 18px;color:#5c564e;background:#faf8f4;font-size:13.5px;font-style:normal;}
.bodywrap code{background:#f0ece4;padding:1px 5px;border-radius:3px;font-size:12.5px;color:#7a3b25;}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

_TAB = {"padding": "13px 20px", "fontFamily": "Inter", "fontSize": "12px",
        "letterSpacing": ".09em", "textTransform": "uppercase", "border": "none",
        "borderBottom": "2px solid transparent", "backgroundColor": "transparent", "color": "#7a736b"}
_TAB_SEL = {**_TAB, "color": "#141414", "borderBottom": "2px solid #bf4e2c", "fontWeight": "600"}


def _tab(label, value):
    return dcc.Tab(label=label, value=value, style=_TAB, selected_style=_TAB_SEL)


_MASTHEAD = html.Div([
    html.Div("For hospital pharmacy procurement · Public-data forecasting", className="eyebrow"),
    html.H1("Drug Shortage Early-Warning Radar"),
    html.Div("Know which drugs run short — months before the shelf does.",
             className="sub"),
], className="masthead")

app.layout = html.Div(html.Div([
    _MASTHEAD,
    html.Div(dcc.Tabs(id="tabs", value="overview", children=[
        _tab("Overview", "overview"),
        _tab("Data & EDA", "eda"),
        _tab("Data Provenance", "provenance"),
        _tab("Watchlist", "watchlist"),
        _tab("Drug Detail", "detail"),
        _tab("Model Performance", "performance"),
        _tab("Benchmark", "benchmark"),
        _tab("Model Card", "model_card"),
    ]), className="tabwrap"),
    html.Div(id="tab-content", className="bodywrap"),
], className="dossier"))


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {
        "overview": _overview_layout,
        "eda": _eda_layout,
        "provenance": _provenance_layout,
        "watchlist": _watchlist_layout,
        "detail": _detail_layout,
        "performance": _performance_layout,
        "benchmark": _benchmark_layout,
        "model_card": _model_card_layout,
    }.get(tab, lambda: html.Div("Select a tab"))()


def _overview_layout():
    panel = FIGDATA.get("panel", pd.DataFrame())
    shortages = FIGDATA.get("shortages", pd.DataFrame())
    test = FIGDATA.get("test", pd.DataFrame())

    n_drugs = panel["drug_key"].nunique() if not panel.empty else 0
    n_onsets = int(shortages["onset_date"].notna().sum()) if not shortages.empty else 0

    roc = p50 = lead = None
    if not test.empty:
        from sklearn.metrics import roc_auc_score
        y = test["y_6m"].astype(int).values
        s = test["risk_score_raw"].values
        roc = roc_auc_score(y, s)
        p50 = F.precision_at_k(y, s, k=50)
        d = test.copy(); d["risk_score"] = d["risk_score_raw"]
        lead = F.lead_time_analysis(d, k=50).get("mean_lead_months")

    cards = dbc.Row([
        _kpi_card(f"{roc:.2f}" if roc else "—", "test ROC-AUC · 2025", accent=True),
        _kpi_card(f"{p50*100:.0f}%" if p50 else "—", "precision @ top-50", accent=True),
        _kpi_card(f"{lead:.1f} mo" if lead else "—", "mean lead time", accent=True),
        _kpi_card(f"{n_drugs:,}", "drugs tracked"),
        _kpi_card(f"{n_onsets:,}", "historical onsets"),
    ], className="g-3")

    return html.Div([
        # ── Hero ──────────────────────────────────────────────────────────────
        html.Div("For the pharmacy buyer · 6-month early warning", className="section-eyebrow"),
        html.H1("A six-month head start on the next shortage.",
                className="section", style={"fontSize": "38px", "maxWidth": "780px",
                                            "margin": "6px 0 10px"}),
        html.P("Each month, the U.S. drugs most likely to go short — ranked by calibrated "
               "risk, with the reason why and ~4 months to line up an alternate. Built from "
               "public FDA data alone.", className="lead", style={"fontSize": "17px"}),
        html.Hr(className="section-rule", style={"marginTop": "18px"}),

        # ── Proof points ──────────────────────────────────────────────────────
        cards,
        html.Div("Headline figures are a one-shot, locked 2025 test — never used for tuning.",
                 className="note warn", style={"marginTop": "14px"}),

        # ── What's inside ─────────────────────────────────────────────────────
        html.Div("What's inside", className="section-eyebrow",
                 style={"marginTop": "30px", "marginBottom": "14px"}),
        dbc.Row([
            _navtile("Drugs to act on", "This month's watchlist, ranked by 6-month risk."),
            _navtile("Drug risk profile", "Why one drug is at risk — score, drivers, history."),
            _navtile("The shortage landscape", "What tends to go short, and why."),
        ], className="g-4 mb-4"),
        dbc.Row([
            _navtile("How far to trust the list", "Accuracy — validation vs. a locked test."),
            _navtile("How it compares", "Versus published shortage-prediction models."),
            _navtile("Method & limits", "How it's built, honestly — and where it can mislead."),
        ], className="g-4"),

        html.Div("Also useful to FDA / public-health analysts, wholesale & GPO planners, and "
                 "researchers — but built first for the buyer.", className="note",
                 style={"marginTop": "26px"}),
    ])


def _grid(figs):
    """Two-column responsive grid of figures."""
    rows = []
    for i in range(0, len(figs), 2):
        rows.append(dbc.Row([dbc.Col(_graph(f), md=6) for f in figs[i:i + 2]], className="g-3 mb-2"))
    return html.Div(rows)


def _eda_layout():
    s, p = FIGDATA["shortages"], FIGDATA["panel"]
    return html.Div([
        _section("Know the terrain", "The shortage landscape",
                 "“What kinds of drugs should I expect to chase — and why?”  Which therapeutic "
                 "categories and causes dominate, how long shortages drag on once they start, "
                 "and why they are so hard to call."),
        _grid([
            F.fig_shortage_trend(s), F.fig_base_rate_over_time(p),
            F.fig_shortage_reasons(s), F.fig_therapeutic_categories(s),
            F.fig_status_donut(s), F.fig_duration_hist(s),
        ]),
        _graph(F.fig_top_shorted_drugs(s)),
    ])


def _performance_layout():
    v, t, imp, feats = FIGDATA["val"], FIGDATA["test"], FIGDATA["importance"], FIGDATA["features"]
    return html.Div([
        _section("Trust check", "How far to trust the list",
                 "“How much should I trust it before I reorder?”  How well it ranks, how often "
                 "the top of the list is right, whether the risk scores are honest probabilities, "
                 "and how early it fires — validation vs. a locked 2025 test (the frozen one-shot)."),
        dcc.Markdown(
            "**How the headline numbers relate.** *ROC-AUC* (0.83 test) measures pure ranking — "
            "can the model sort risky from safe drugs. *PR-AUC* (0.11) looks bleak only because "
            "shortages are ~2% rare; it averages over the hopeless long tail. What the tool "
            "actually delivers is the **operating point**: at a top-50 monthly watchlist, "
            "**precision@50 = 24%** on test — about 1 in 4 flagged drugs truly go short within "
            "6 months (~11× the base rate). The ✦ marker on the PR and precision@K charts below "
            "shows exactly where that ship-point sits on each curve.", className="note"),
        _graph(F.fig_val_test_metrics(v, t)),
        _grid([
            F.fig_pr_curve(v, t), F.fig_roc_curve(v, t),
            F.fig_precision_at_k(v, t), F.fig_lead_time_hist(v, t),
            F.fig_calibration(v, t), F.fig_score_distribution(v),
            F.fig_feature_importance(imp), F.fig_shap_summary(feats),
        ]),
    ])


_BENCH_TABLE = (
    "| Study | Data | Horizon | Headline | Operating point |\n"
    "|---|---|---|---|---|\n"
    "| UC Berkeley 2024 | Public FDA (same as ours) | 4 weeks | AUROC 0.93 | recall 72% / **precision 0.1%** |\n"
    "| Canadian XGBoost 2023 | Proprietary pharmacy sales | 1 month | accuracy 69%, κ 0.44 | 59% recall on severe |\n"
    "| South Korea 2025 | Regulatory case reports | duration / cause | F1 > 0.70 | classifies cause, not onset |\n"
    "| **This model** | Public FDA + Wayback | **6 months** | ROC-AUC 0.83 | **precision@50 = 24%** |\n"
)


def _benchmark_layout():
    return html.Div([
        _section("Versus the status quo", "How it compares",
                 "“Is this better than waiting for the FDA to post?”  Most teams find out a drug "
                 "is short when the FDA posts it — too late to plan. This gives a ranked watchlist "
                 "months earlier. Below is how it compares to published shortage-prediction models "
                 "(no study reports precision@K at this horizon, so the fair common metric is AUROC)."),
        _grid([F.fig_benchmark_positioning(), F.fig_benchmark_metrics()]),
        dcc.Markdown(_BENCH_TABLE),
        html.Hr(className="section-rule"),
        dbc.Row([
            _takeaway("Same data, opposite corner",
                      "A near-identical public-data peer (Berkeley) reports 0.1% precision at "
                      "72% recall — a high AUC that is operationally unusable. We optimise the "
                      "usable corner: 24% precision in the top-50."),
            _takeaway("Higher numbers ≠ better method",
                      "Stronger headlines come from easier 1-month horizons, proprietary "
                      "transaction data, or rare-event-flattering metrics — not better "
                      "modelling. Even with pharmacy-sales data, κ tops out at 0.44."),
            _takeaway("The ceiling is the data",
                      "Across every study the dominant signal is prior-shortage history. The "
                      "limit here is the public-data feature set, not the algorithm."),
        ], className="g-3"),
        html.P("Sources: UC Berkeley I-School project (2024); Health Care Management Science "
               "10.1007/s10729-022-09627-y; Frontiers in Pharmacology 10.3389/fphar.2025.1608843.",
               className="note", style={"marginTop": "20px"}),
    ])


def _watchlist_layout():
    route_options = []
    if not FEAT.empty and "drug_key" in FEAT.columns:
        routes = sorted(FEAT["drug_key"].str.split("|").str[1].dropna().unique().tolist())
        route_options = [{"label": r.title(), "value": r} for r in routes]

    return html.Div([
        _section("Your monthly list", "Drugs to act on",
                 "“Which drugs should I act on right now?”  Drugs not currently short, ranked by "
                 "calibrated 6-month onset risk — the ones to pre-order, build par stock on, or "
                 "line up a second supplier for. Filter by route; widen the list with Top-N."),
        html.Div([
            html.Label("Filter by route:"),
            dcc.Dropdown(id="route-filter", options=route_options, multi=True,
                         placeholder="All routes", style={"width": "300px"}),
            html.Label("Top N:", style={"marginLeft": "20px"}),
            dcc.Slider(id="topn-slider", min=10, max=100, step=10, value=50,
                       marks={10: "10", 50: "50", 100: "100"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "10px"}),
        html.Div(id="watchlist-table"),
    ])


def _detail_layout():
    drug_options = []
    # List every drug that actually has predictions (train+val), not just the first 500.
    src = PREDS if not PREDS.empty and "drug_key" in PREDS.columns else FEAT
    if not src.empty and "drug_key" in src.columns:
        keys = sorted(src["drug_key"].unique().tolist())
        drug_options = [{"label": _drug_key_to_label(k), "value": k} for k in keys]

    return html.Div([
        _section("One drug, one decision", "Drug risk profile",
                 f"“Should I act on this drug — and what’s driving it?”  Its risk trajectory, "
                 f"the factors pushing the score up or down, and its shortage history — enough "
                 f"to decide whether to reorder, qualify an alternate, or hold. "
                 f"Searchable across {len(drug_options):,} drugs."),
        dcc.Dropdown(id="drug-selector", options=drug_options, maxHeight=320,
                     optionHeight=44, searchable=True, clearable=True,
                     placeholder="Type a drug name to search…",
                     style={"width": "560px", "marginBottom": "20px"}),
        html.Div(id="drug-detail-content"),
    ])


def _provenance_layout():
    sources_md = (
        "| Dataset | Source | Role |\n"
        "|---|---|---|\n"
        "| **FDA Drug Shortages** | openFDA live API + **Wayback Machine** captures of FDA shortage pages | **The label** — reconstructed history of onsets & resolutions |\n"
        "| NDC Directory | openFDA bulk | Drug universe, dosage forms, manufacturers |\n"
        "| Drugs@FDA | openFDA bulk | Approvals → generic-competitor counts, drug age |\n"
        "| Drug Enforcement / Recalls | openFDA bulk | `months_since_last_recall` |\n"
        "| Inspections / Classifications | FDA Data Dashboard API | `months_since_last_inspection` |\n"
        "| Compliance / Citations | FDA Data Dashboard API | Built, then dropped (2011–2022 coverage gap) |\n"
        "| NADAC pricing | CMS Medicaid open data | Drug acquisition cost signal |\n"
    )
    return html.Div([
        _section("Where the data comes from", "Data provenance",
                 "“Can I trust the inputs?”  Everything here is public U.S. federal data. The "
                 "hardest part wasn’t modeling — it was recovering shortage history the FDA "
                 "does not keep."),
        html.Div("The catch: the FDA shortage feed (openFDA + the live website) is a "
                 "current snapshot only. When a shortage resolves, the record is purged — so "
                 "there is no official history of when past shortages started or ended. But the "
                 "shortage history is exactly what the model has to learn from.",
                 className="note warn"),
        html.H4("Reconstructing the label from the Internet Archive", className="section",
                style={"fontSize": "19px", "marginTop": "8px"}),
        html.P("To rebuild the timeline, the pipeline pulls dated captures of the FDA shortage "
               "pages from the Wayback Machine and stitches them into a month-by-month record "
               "of what was on the list — from which onset and resolution dates are derived.",
               className="lead"),
        html.Ul([
            html.Li("Wayback CDX API lists every archived snapshot of the FDA shortage pages."),
            html.Li("HTML listing page: ~2,400 captures back to 2014 (dense from 2020)."),
            html.Li("Downloadable CSV version: ~96 captures from 2019-10 onward."),
            html.Li("Raw original page is fetched (Wayback `id_` mode), parsed for drug name, "
                    "status, and dates, then deduplicated into a continuous shortage history."),
        ]),
        html.Hr(className="section-rule"),
        html.H4("All sources at a glance", className="section", style={"fontSize": "19px"}),
        dcc.Markdown(sources_md),
        html.Div("Honest caveat: Wayback coverage is thin before ~2014 and densest 2020+, so "
                 "the reconstructed history is most reliable in recent years — which is why the "
                 "train / validation / locked-test windows sit where they do. Shortages that "
                 "began and resolved entirely within a snapshot gap are missed.",
                 className="note"),
    ])


def _model_card_layout():
    val_path = REPORTS_DIR / "validation.md"
    test_path = REPORTS_DIR / "test_LOCKED.md"
    val_text = val_path.read_text() if val_path.exists() else \
        "Validation report not yet generated. Run scripts/evaluate.py first."
    test_text = test_path.read_text() if test_path.exists() else ""

    v, t = FIGDATA["val"], FIGDATA["test"]
    children = [_section("Before you rely on it", "Method & limits",
                         "“What are the limits before I lean on this?”  How the risk is built, "
                         "the honest validation and locked-test results, and — most importantly "
                         "for a buyer — where it can mislead, so you know when to trust the list "
                         "and when to pick up the phone.")]
    if test_text:
        children += [html.Div("Locked test was run once on the frozen model — the honest "
                              "out-of-sample estimate.", className="note warn"),
                     dcc.Markdown(test_text),
                     _grid([F.fig_val_test_metrics(v, t), F.fig_lead_time_hist(v, t)]),
                     html.Hr(className="section-rule")]
    children += [dcc.Markdown(val_text), html.Hr(className="section-rule")]
    return html.Div(children + [
        html.H4("Known limitations", className="section", style={"fontSize": "19px"}),
        html.Ul([
            html.Li("Entity resolution: ~60–80% of inspections linked to drugs; imperfect linkage attenuates manufacturing-risk signal."),
            html.Li("Label = FDA posting date, not true supply disruption onset (FDA typically posts 1–4 weeks after disruption begins)."),
            html.Li("Wayback Machine gaps: shortages that began and resolved within a snapshot gap (~monthly) are missed, mainly before 2020."),
            html.Li("NDC marketing status is a snapshot: drugs that exited market without updating NDC end-date may persist in panel."),
            html.Li("This tool is a triage aid, not an oracle. False alarms can themselves distort purchasing; use in conjunction with domain expertise."),
        ]),
    ])


# ── Watchlist callback ────────────────────────────────────────────────────────

@app.callback(
    Output("watchlist-table", "children"),
    Input("route-filter", "value"),
    Input("topn-slider", "value"),
)
def update_watchlist(routes, top_n):
    return _compute_watchlist(routes, top_n)


def _compute_watchlist(routes, top_n):
    """Pure function — testable without browser."""
    if PREDS.empty:
        return html.P("No predictions available. Run scripts/train.py first.")

    preds = _latest_predictions("val")
    if preds.empty:
        return html.P("No validation predictions found.")

    # Filter out drugs currently in shortage
    preds = preds[~preds["drug_key"].str.split("|").str[0].isin(CURRENTLY_SHORT)]

    # Filter by route
    if routes:
        preds = preds[preds["drug_key"].str.split("|").str[1].isin(routes)]

    top = preds.head(top_n).copy()
    top["Drug"] = top["drug_key"].apply(_drug_key_to_label)
    top["Route"] = top["drug_key"].str.split("|").str[1].str.title()
    top["Risk Score"] = top["risk_score"].round(3)
    top["Went Short (Val)"] = top["y_6m"].apply(lambda x: "✓" if x else "")
    top["Rank"] = range(1, len(top) + 1)

    display_cols = ["Rank", "Drug", "Route", "Risk Score", "Went Short (Val)"]
    table_data = top[display_cols].to_dict("records")

    return dash_table.DataTable(
        id="watchlist-datatable",
        data=table_data,
        columns=[{"name": c, "id": c} for c in display_cols],
        style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "Inter, sans-serif"},
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "600"},
        style_data_conditional=[
            {"if": {"filter_query": "{Risk Score} > 0.3"}, "backgroundColor": "#fbe3e0", "color": "#a31515"},
            {"if": {"filter_query": "{Risk Score} > 0.15 && {Risk Score} <= 0.3"}, "backgroundColor": "#fdf3df"},
            {"if": {"column_id": "Went Short (Val)", "filter_query": "{Went Short (Val)} = '✓'"},
             "color": "#c0392b", "fontWeight": "bold"},
        ],
        page_size=25,
        sort_action="native",
    )


# ── Drug detail callbacks ─────────────────────────────────────────────────────

@app.callback(
    Output("drug-detail-content", "children"),
    Input("drug-selector", "value"),
)
def update_drug_detail(drug_key):
    return _compute_drug_detail(drug_key)


def _compute_drug_detail(drug_key):
    """Pure function — testable without browser."""
    if not drug_key:
        return html.P("Select a drug to see its risk profile.")

    if PREDS.empty or "drug_key" not in PREDS.columns:
        return html.P("No predictions available.")

    drug_preds = PREDS[PREDS["drug_key"] == drug_key].sort_values("month")
    if drug_preds.empty:
        return html.P(f"No prediction history for {_drug_key_to_label(drug_key)}")

    latest_score = float(drug_preds.iloc[-1]["risk_score"])

    # Risk score sparkline (dossier theme)
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=drug_preds["month"], y=drug_preds["risk_score"] * 100,
        mode="lines", name="Risk", line={"color": F.INK, "width": 2},
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    ))
    if "y_6m" in drug_preds.columns:
        for sm in drug_preds[drug_preds["y_6m"] == True]["month"]:
            sm_str = pd.Timestamp(sm).isoformat()
            fig_risk.add_shape(type="line", x0=sm_str, x1=sm_str, y0=0, y1=1,
                               xref="x", yref="paper",
                               line={"color": F.ACCENT, "dash": "dot", "width": 1.5})
        fig_risk.add_annotation(x=0.99, y=0.97, xref="paper", yref="paper", showarrow=False,
                                xanchor="right", text="┊ shortage onset", font={"color": F.ACCENT, "size": 11})
    fig_risk = F._style(fig_risk, "Risk-score history", height=250,
                        yaxis_title="Risk (%)", xaxis_title=None)

    # Top SHAP risk drivers for the most recent month (Phase 7 explain integration).
    # Computed lazily + defensively: a SHAP/import failure must not break the tab.
    drivers_block = html.Div()
    try:
        from src.explain import top_drivers
        latest_month = drug_preds["month"].max()
        feat_src = FEAT if not FEAT.empty else PREDS
        drivers = top_drivers(feat_src, drug_key, latest_month, top_n=5)
        if drivers:
            driver_rows = [{
                "Driver": d["feature"],
                "Value": round(d["feature_value"], 2),
                "Effect on risk": ("▲ raises" if d["shap_value"] > 0 else "▼ lowers"),
                "SHAP": round(d["shap_value"], 3),
            } for d in drivers]
            drivers_block = html.Div([
                html.H5("Why — top risk drivers (latest month)"),
                dash_table.DataTable(
                    data=driver_rows,
                    columns=[{"name": c, "id": c} for c in ["Driver", "Value", "Effect on risk", "SHAP"]],
                    style_cell={"textAlign": "left", "padding": "6px"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{SHAP} > 0"}, "color": "#ff6b6b"},
                        {"if": {"filter_query": "{SHAP} < 0"}, "color": "#2ecc71"},
                    ],
                ),
            ])
    except Exception as e:  # noqa: BLE001 — explanation is best-effort
        drivers_block = html.P(f"(Driver breakdown unavailable: {e})",
                               style={"color": "#888", "fontStyle": "italic"})

    # Shortage history
    generic = drug_key.split("|")[0]
    drug_shortages = SHORTAGES[
        SHORTAGES["generic_name"].str.lower().str.strip() == generic
    ] if not SHORTAGES.empty and "generic_name" in SHORTAGES.columns else pd.DataFrame()

    shortage_rows = []
    if not drug_shortages.empty:
        for _, row in drug_shortages.iterrows():
            shortage_rows.append({
                "Onset Date": str(row.get("onset_date", ""))[:10],
                "Company": row.get("company_name", ""),
                "Reason": row.get("shortage_reason", ""),
                "Status": row.get("status_last", ""),
            })

    # Current live status
    live_status = "Not in current FDA shortage list"
    if not LIVE.empty and "generic_name" in LIVE.columns:
        live_match = LIVE[LIVE["generic_name"].str.lower().str.strip() == generic]
        if not live_match.empty:
            statuses = live_match["status"].unique().tolist() if "status" in live_match.columns else []
            live_status = f"CURRENT FDA STATUS: {', '.join(statuses)}"

    is_current = "Current" in live_status
    return html.Div([
        html.Div([
            html.H4(_drug_key_to_label(drug_key), className="section", style={"fontSize": "25px", "margin": 0}),
            html.Span(live_status, style={
                "color": "#9b2d1f" if is_current else "#2e7d32", "fontSize": "12.5px",
                "fontWeight": "600", "letterSpacing": ".02em"}),
        ], style={"marginBottom": "16px"}),
        _risk_readout(latest_score),
        dcc.Graph(figure=fig_risk, config={"displayModeBar": False}),
        drivers_block,
        html.H5("Shortage history", className="section", style={"fontSize": "19px", "marginTop": "20px"}),
        (dash_table.DataTable(
            data=shortage_rows,
            columns=[{"name": c, "id": c} for c in ["Onset Date", "Company", "Reason", "Status"]],
            style_cell={"textAlign": "left", "padding": "8px 10px", "fontFamily": "Inter, sans-serif",
                        "fontSize": "13px", "border": "none", "borderBottom": "1px solid #e2ddd5"},
            style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "600",
                          "textTransform": "uppercase", "fontSize": "11px"},
            page_size=10,
        ) if shortage_rows else html.P("No shortage history found in dataset.", style={"color": "#7a736b"})),
        html.Hr(className="section-rule", style={"marginTop": "26px"}),
        html.Div([
            html.Div("Methodology", className="section-eyebrow"),
            dcc.Markdown(_RISK_EXPLAINER),
        ], className="note", style={"marginTop": "6px"}),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
