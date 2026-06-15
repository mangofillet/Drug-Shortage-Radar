"""
Figure-builder tests — confirm every builder returns a populated go.Figure without a
browser (matches the project's no-Selenium rule). Uses real cached parquet data; skips
gracefully if the pipeline hasn't been built yet.
"""
import sys
from pathlib import Path

import pytest
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import figures as F


@pytest.fixture(scope="module")
def data():
    d = F.load_all()
    if d["shortages"].empty or d["val"].empty:
        pytest.skip("Pipeline outputs missing — run build_panel/train/build_figures first")
    return d


EDA_BUILDERS = ["fig_shortage_trend", "fig_shortage_reasons", "fig_therapeutic_categories",
                "fig_status_donut", "fig_duration_hist", "fig_top_shorted_drugs"]


@pytest.mark.parametrize("name", EDA_BUILDERS)
def test_eda_builders(data, name):
    fig = getattr(F, name)(data["shortages"])
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_base_rate(data):
    fig = F.fig_base_rate_over_time(data["panel"])
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


PERF_PAIR = ["fig_pr_curve", "fig_roc_curve", "fig_calibration",
             "fig_precision_at_k", "fig_lead_time_hist", "fig_val_test_metrics"]


@pytest.mark.parametrize("name", PERF_PAIR)
def test_perf_val_test_builders(data, name):
    fig = getattr(F, name)(data["val"], data["test"])
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_importance_and_score_dist(data):
    assert len(F.fig_feature_importance(data["importance"]).data) >= 1
    assert len(F.fig_score_distribution(data["val"]).data) >= 1


def test_shap_summary(data):
    fig = F.fig_shap_summary(data["features"], sample=300)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_benchmark_builders():
    assert len(F.fig_benchmark_positioning().data) >= 1
    assert len(F.fig_benchmark_metrics().data) >= 1


def test_empty_input_is_safe():
    import pandas as pd
    # Builders must not crash on empty frames — they return a placeholder figure.
    assert isinstance(F.fig_shortage_trend(pd.DataFrame()), go.Figure)
    assert isinstance(F.fig_pr_curve(pd.DataFrame(), pd.DataFrame()), go.Figure)
