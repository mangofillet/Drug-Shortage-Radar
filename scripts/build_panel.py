"""
Phase 3+4: builds drug-month panel, assigns labels, engineers features.
Usage: python scripts/build_panel.py [--force]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resolve import build_firm_link
from src.panel import build_panel
from src.features import build_features


def main(force: bool = False):
    print("=" * 60)
    print("PHASE 2: ENTITY RESOLUTION")
    print("=" * 60)
    firm_link = build_firm_link(force=force)

    print("\n" + "=" * 60)
    print("PHASE 3: PANEL + LABELS")
    print("=" * 60)
    panel = build_panel(force=force)

    # Merge split column onto features
    print("\n" + "=" * 60)
    print("PHASE 4: FEATURES")
    print("=" * 60)
    # Panel has split; pass it to features
    features = build_features(panel, force=force)
    # Merge split from panel
    features = features.merge(
        panel[["drug_key", "month", "split", "y_6m", "y_3m", "next_onset_month"]],
        on=["drug_key", "month"],
        how="left",
        suffixes=("_feat", ""),
    )
    # Resolve duplicate y_6m columns
    if "y_6m_feat" in features.columns:
        features["y_6m"] = features["y_6m"].fillna(features["y_6m_feat"])
        features = features.drop(columns=["y_6m_feat"])
    # Drop leftover *_feat duplicates from the merge (split_feat, y_3m_feat) so the
    # canonical split / y_3m / next_onset_month columns (from the panel) are the only ones.
    dup_feat = [c for c in features.columns if c.endswith("_feat")]
    if dup_feat:
        features = features.drop(columns=dup_feat)

    from config import DATA_PROCESSED
    out = DATA_PROCESSED / "features_with_labels.parquet"
    features.to_parquet(out, index=False)
    print(f"\nFinal feature matrix → {out}")
    print(f"  Shape: {features.shape}")

    for split in ["train", "val", "test"]:
        sp = features[features["split"] == split]
        if len(sp) == 0:
            continue
        rate = sp["y_6m"].mean() * 100 if sp["y_6m"].notna().any() else 0
        print(f"  {split}: {len(sp):,} rows, {rate:.2f}% positive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
