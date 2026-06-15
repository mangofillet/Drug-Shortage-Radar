"""
Phase 5 — train + validate models, then persist scored predictions.
Never touches the test window.
Usage: python scripts/train.py [--skip-cv]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import DATA_PROCESSED
from src.models import run_training, save_predictions


def main(skip_cv: bool = False):
    feat_path = DATA_PROCESSED / "features_with_labels.parquet"
    if not feat_path.exists():
        print("ERROR: Run scripts/build_panel.py first")
        sys.exit(1)

    features = pd.read_parquet(feat_path)
    print(f"Loaded features: {features.shape}")

    run_training(features, skip_cv=skip_cv)

    print("\n── Scoring predictions (train+val; test stays locked) ──")
    save_predictions(features)

    print("\nTraining complete. Run scripts/evaluate.py for the validation report.")
    print("REMINDER: Do NOT run eval_test_LOCKED.py until the user explicitly requests it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true", help="reuse cached CV best_iter")
    args = parser.parse_args()
    main(skip_cv=args.skip_cv)
