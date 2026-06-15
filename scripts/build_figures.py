"""
Export every dashboard figure to reports/figures/*.png (kaleido) + a combined
reports/figures.html, for embedding in the README / written capstone report.

Ensures predictions_test.parquet exists first — this DISPLAYS the already-spent,
frozen one-shot test result; it does NOT retrain or tune anything.
Usage: python scripts/build_figures.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd

from config import DATA_PROCESSED, REPORTS
from src.figures import load_all, build_all

FIG_DIR = REPORTS / "figures"

# Titles for the combined HTML contact sheet
SECTIONS = {
    "Data & EDA": ["eda_shortage_trend", "eda_base_rate", "eda_reasons", "eda_categories",
                   "eda_status", "eda_duration", "eda_top_drugs"],
    "Model performance (validation vs. locked test)":
        ["perf_val_test", "perf_pr_curve", "perf_roc_curve", "perf_calibration",
         "perf_precision_at_k", "perf_lead_time", "perf_importance", "perf_shap",
         "perf_score_dist"],
    "Benchmark vs. prior work": ["bench_positioning", "bench_metrics"],
}


def _ensure_test_predictions():
    if (DATA_PROCESSED / "predictions_test.parquet").exists():
        return
    print("predictions_test.parquet missing — scoring frozen test split (DISPLAY ONLY)…")
    from src.models import save_test_predictions
    feats = pd.read_parquet(DATA_PROCESSED / "features_with_labels.parquet")
    save_test_predictions(feats)


def main():
    print("NOTE: test figures show the FROZEN one-shot 2025 result — display only, no tuning.\n")
    _ensure_test_predictions()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = build_all(load_all())

    n_png = 0
    for name, fig in figs.items():
        try:
            fig.write_image(str(FIG_DIR / f"{name}.png"), width=900, height=fig.layout.height or 400, scale=2)
            n_png += 1
        except Exception as e:  # noqa: BLE001
            print(f"  PNG failed for {name}: {e}")
    print(f"Wrote {n_png}/{len(figs)} PNGs → {FIG_DIR}")

    # Combined HTML contact sheet
    parts = ["<html><head><meta charset='utf-8'><title>Drug Shortage Radar — Figures</title>",
             "<style>body{font-family:Inter,Segoe UI,sans-serif;max-width:1000px;margin:24px auto;"
             "color:#222;padding:0 16px}h1{font-size:24px}h2{margin-top:36px;border-bottom:2px solid #eee;"
             "padding-bottom:6px}</style></head><body>",
             "<h1>Drug Shortage Radar — figure pack</h1>"]
    import plotly.io as pio
    first = True
    for section, names in SECTIONS.items():
        parts.append(f"<h2>{section}</h2>")
        for name in names:
            if name in figs:
                parts.append(pio.to_html(figs[name], include_plotlyjs="cdn" if first else False,
                                         full_html=False))
                first = False
    parts.append("</body></html>")
    html_path = REPORTS / "figures.html"
    html_path.write_text("\n".join(parts))
    print(f"Wrote contact sheet → {html_path}")


if __name__ == "__main__":
    main()
