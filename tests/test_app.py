"""
Phase 7 gate: callback unit tests + Flask test-client smoke test.
No Selenium, no dash[testing], no Chrome.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as app_module
from app import _compute_watchlist, _compute_drug_detail, app


def test_app_layout_has_key_ids():
    layout = app.layout
    layout_str = str(layout)
    assert "tabs" in layout_str
    assert "tab-content" in layout_str


import pytest
from app import render_tab


@pytest.mark.parametrize("tab", ["overview", "eda", "watchlist", "detail",
                                  "performance", "benchmark", "model_card"])
def test_every_tab_renders(tab):
    comp = render_tab(tab)
    assert comp is not None


def test_compute_watchlist_empty_preds(monkeypatch):
    monkeypatch.setattr(app_module, "PREDS", pd.DataFrame())
    result = _compute_watchlist(routes=None, top_n=50)
    assert result is not None


def test_compute_watchlist_with_data(monkeypatch):
    sample_preds = pd.DataFrame({
        "drug_key": ["vancomycin|injection", "amoxicillin|oral"],
        "month": [pd.Timestamp("2024-01-01")] * 2,
        "split": ["val", "val"],
        "y_6m": [True, False],
        "risk_score": [0.8, 0.2],
    })
    monkeypatch.setattr(app_module, "PREDS", sample_preds)
    monkeypatch.setattr(app_module, "CURRENTLY_SHORT", set())
    result = _compute_watchlist(routes=None, top_n=50)
    assert result is not None


def test_compute_drug_detail_no_selection():
    result = _compute_drug_detail(None)
    assert result is not None


def test_compute_drug_detail_with_drug(monkeypatch):
    sample_preds = pd.DataFrame({
        "drug_key": ["vancomycin|injection"] * 3,
        "month": pd.date_range("2023-01-01", periods=3, freq="MS"),
        "split": ["val"] * 3,
        "y_6m": [False, False, True],
        "risk_score": [0.1, 0.3, 0.7],
    })
    monkeypatch.setattr(app_module, "PREDS", sample_preds)
    monkeypatch.setattr(app_module, "SHORTAGES", pd.DataFrame())
    monkeypatch.setattr(app_module, "LIVE", pd.DataFrame())
    result = _compute_drug_detail("vancomycin|injection")
    assert result is not None


def test_flask_client_get_root():
    client = app.server.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"<!DOCTYPE html" in r.data.lower() or b"<html" in r.data.lower()


def test_flask_client_dash_layout():
    client = app.server.test_client()
    r = client.get("/_dash-layout")
    assert r.status_code == 200


def test_flask_client_dash_dependencies():
    client = app.server.test_client()
    r = client.get("/_dash-dependencies")
    assert r.status_code == 200
