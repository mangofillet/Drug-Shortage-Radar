"""
Phase 3 — Drug-month panel construction, label assignment, temporal splits.
"""

import pandas as pd
import numpy as np

from config import (
    DATA_PROCESSED, STUDY_START, TRAIN_END, VAL_START,
    TEST_START, TEST_END, LABEL_HORIZON_MONTHS, LABEL_HORIZON_SHORT_MONTHS,
    MARKETING_CATEGORIES,
)


import re as _re

# Dosage form / route keywords that appear at the END of FDA shortage generic names
# but are NOT part of the NDC generic_name field.
# Intentionally excludes salt forms (hydrochloride, sulfate, sodium, etc.) — those ARE INN names.
_DOSAGE_FORM_RE = _re.compile(
    r"[\s,]+"
    r"(injection|injectable|infusion|intravenous|iv\b|oral\b|tablet|tablets|capsule|capsules|"
    r"solution|suspension|syrup|elixir|cream|ointment|gel|lotion|patch|film|spray|"
    r"inhaler|inhalation|powder|drops|drop|suppository|suppositories|implant|pellet|"
    r"concentrate|kit|vial|ampoule|ampule|syringe|prefilled|lyophilized|"
    r"extended.release|immediate.release|modified.release|delayed.release|sustained.release|"
    r"ophthalmic|otic|nasal|rectal|topical|transdermal|sublingual|buccal|"
    r"for\s+injection|for\s+infusion|preservative.free|"
    r"\d+\s*mg[\s,].*|\d+\s*mcg[\s,].*|\d+\s*ml[\s,].*|\d+\s*%[\s,].*|"
    r"\d+\s*mg$|\d+\s*mcg$|\d+\s*ml$|\d+\s*%$)"
    r"[\s,]*(.*)?$",
    _re.IGNORECASE,
)
# Also strip parenthetical brand names e.g. "Lidocaine Hydrochloride (Xylocaine)"
_PAREN_RE = _re.compile(r"\s*\(.*?\)")


def _normalize_shortage_name(name: str) -> str:
    """Strip dosage form and parenthetical brand names from shortage generic_name."""
    if not name:
        return ""
    s = _PAREN_RE.sub("", name).strip()
    s = _DOSAGE_FORM_RE.sub("", s).strip().rstrip(",").strip()
    return s.lower().strip()


def _tokset(name: str) -> frozenset:
    """
    Order-independent ingredient signature: the set of word-tokens (>=4 chars) in a
    name. Combination drugs (e.g. Adderall's four salts) are listed in different orders
    and separators across FDA shortage records vs the NDC directory, so exact-string
    matching misses them; comparing token sets recovers those links without the false
    positives of fuzzy/substring matching (we require the full ingredient set to match).
    """
    if not name:
        return frozenset()
    return frozenset(_re.findall(r"[a-z]{4,}", name.lower()))


def _parse_route(route_str) -> str:
    if not route_str or pd.isna(route_str):
        return ""
    # Value stored as "['ORAL']" or "['INTRA-ARTERIAL', 'INTRAVENOUS']"
    return str(route_str).strip("[]'\" ").split(",")[0].strip("'\" ").lower()


def build_panel(force: bool = False) -> pd.DataFrame:
    out_path = DATA_PROCESSED / "panel.parquet"
    if out_path.exists() and not force:
        print(f"Using cached panel: {out_path}")
        return pd.read_parquet(out_path)

    ndc = pd.read_parquet(DATA_PROCESSED / "ndc.parquet")
    shortages = pd.read_parquet(DATA_PROCESSED / "shortages_raw.parquet")

    # --- Drug universe ---
    rx = ndc[ndc["marketing_category"].isin(MARKETING_CATEGORIES)].copy()
    rx["route_norm"] = rx["route"].apply(_parse_route)
    rx["generic_norm"] = rx["generic_name"].str.lower().str.strip().fillna("")
    rx["drug_key"] = rx["generic_norm"] + "|" + rx["route_norm"]

    study_start = pd.Timestamp(STUDY_START)
    study_end = pd.Timestamp(TEST_END) + pd.offsets.MonthEnd(0)

    drug_dates = (
        rx.groupby("drug_key")
        .agg(
            first_marketed=("marketing_start_date", "min"),
            last_marketed=("marketing_end_date", "max"),
            n_ndc=("product_ndc", "nunique"),
            n_labelers=("labeler_name", "nunique"),
        )
        .reset_index()
    )
    drug_dates["first_marketed"] = pd.to_datetime(drug_dates["first_marketed"], errors="coerce")
    drug_dates["last_marketed"] = pd.to_datetime(drug_dates["last_marketed"], errors="coerce")

    # --- Build panel skeleton ---
    months = pd.date_range(study_start, study_end, freq="MS")
    print(f"Building panel: {len(drug_dates):,} drug_keys × {len(months)} months...")

    rows = []
    for _, drug in drug_dates.iterrows():
        dk = drug["drug_key"]
        fm = drug["first_marketed"]
        lm = drug["last_marketed"]
        start = max(study_start, fm) if pd.notna(fm) else study_start
        end = lm if pd.notna(lm) else study_end
        drug_months = months[(months >= start) & (months <= end)]
        if len(drug_months):
            rows.extend((dk, m) for m in drug_months)

    panel = pd.DataFrame(rows, columns=["drug_key", "month"])
    print(f"  Panel skeleton: {len(panel):,} rows")

    # --- Shortage onset events ---
    shortage_onsets = shortages[shortages["onset_date"].notna()].copy()
    shortage_onsets["generic_norm"] = shortage_onsets["generic_name"].str.lower().str.strip()
    shortage_onsets["onset_month"] = shortage_onsets["onset_date"].dt.to_period("M").dt.to_timestamp()

    # Map generic_norm → list of drug_keys (route-agnostic — same drug may have oral + injectable)
    panel_dk_by_generic = (
        pd.Series(panel["drug_key"].unique())
        .to_frame("drug_key")
        .assign(generic_norm=lambda d: d["drug_key"].str.split("|").str[0])
        .groupby("generic_norm")["drug_key"]
        .apply(list)
        .to_dict()
    )
    # Secondary index: ingredient token-set → drug_keys, for combination drugs whose
    # salts are listed in a different order/separator in the shortage feed than in NDC.
    panel_dk_by_tokset: dict[frozenset, list] = {}
    for gn, keys in panel_dk_by_generic.items():
        ts = _tokset(gn)
        if ts:
            panel_dk_by_tokset.setdefault(ts, []).extend(keys)

    # Explode to (drug_key, onset_month) pairs
    # Use stripped shortage name for matching (shortage names include dosage form)
    n_exact = n_tokset = 0
    onset_pairs = []
    for _, row in shortage_onsets.iterrows():
        gn_stripped = _normalize_shortage_name(row["generic_norm"])
        # Try stripped name, then original, then order-independent token-set match.
        keys = panel_dk_by_generic.get(gn_stripped) or panel_dk_by_generic.get(row["generic_norm"])
        if keys:
            n_exact += 1
        else:
            keys = panel_dk_by_tokset.get(_tokset(gn_stripped), [])
            if keys:
                n_tokset += 1
        for dk in keys:
            onset_pairs.append((dk, row["onset_month"]))
    print(f"  Shortage→drug matches: {n_exact} exact, {n_tokset} via token-set")

    onsets_df = (
        pd.DataFrame(onset_pairs, columns=["drug_key", "onset_month"])
        .drop_duplicates()
        .assign(onset_month=lambda d: pd.to_datetime(d["onset_month"]))
    )
    print(f"  Shortage onset pairs: {len(onsets_df):,}")

    # --- Label expansion (vectorized) ---
    # For each onset, expand backward: months in [onset - horizon, onset) get y_6m=True
    # For in_shortage: months in [onset, onset + 12m) get in_shortage=True
    H6 = LABEL_HORIZON_MONTHS
    H3 = LABEL_HORIZON_SHORT_MONTHS

    label_rows_6m = []
    label_rows_3m = []
    in_shortage_rows = []

    for _, row in onsets_df.iterrows():
        dk = row["drug_key"]
        onset = row["onset_month"]
        # Label window: H6 months BEFORE onset
        for delta in range(1, H6 + 1):
            m = onset - pd.DateOffset(months=delta)
            label_rows_6m.append({"drug_key": dk, "month": m, "next_onset_month": onset})
            if delta <= H3:
                label_rows_3m.append({"drug_key": dk, "month": m})
        # In-shortage window: 12 months from onset
        for delta in range(0, 12):
            m = onset + pd.DateOffset(months=delta)
            in_shortage_rows.append({"drug_key": dk, "month": m})

    def _to_month_start(df):
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
        return df

    y6m_set = _to_month_start(pd.DataFrame(label_rows_6m)).drop_duplicates(["drug_key", "month"]) if label_rows_6m else pd.DataFrame(columns=["drug_key", "month", "next_onset_month"])
    y3m_set = _to_month_start(pd.DataFrame(label_rows_3m)).drop_duplicates() if label_rows_3m else pd.DataFrame(columns=["drug_key", "month"])
    in_shortage_set = _to_month_start(pd.DataFrame(in_shortage_rows)).drop_duplicates() if in_shortage_rows else pd.DataFrame(columns=["drug_key", "month"])

    # For drugs with multiple onsets, keep earliest next_onset_month per (drug_key, month)
    if len(y6m_set) > 0:
        y6m_set = (
            y6m_set.groupby(["drug_key", "month"], as_index=False)
            .agg(next_onset_month=("next_onset_month", "min"))
        )

    # --- Join labels to panel ---
    panel["month"] = pd.to_datetime(panel["month"])

    panel = panel.merge(
        y6m_set.assign(y_6m=True),
        on=["drug_key", "month"], how="left"
    )
    panel = panel.merge(
        y3m_set.assign(y_3m=True),
        on=["drug_key", "month"], how="left"
    )
    panel = panel.merge(
        in_shortage_set.assign(in_shortage=True),
        on=["drug_key", "month"], how="left"
    )

    panel["y_6m"] = panel["y_6m"].fillna(False).astype(bool)
    panel["y_3m"] = panel["y_3m"].fillna(False).astype(bool)
    panel["in_shortage"] = panel["in_shortage"].fillna(False).astype(bool)

    # --- Splits and cleanup ---
    panel["split"] = "train"
    panel.loc[panel["month"] >= pd.Timestamp(VAL_START), "split"] = "val"
    panel.loc[panel["month"] >= pd.Timestamp(TEST_START), "split"] = "test"

    # Remove rows where drug is actively in shortage (predicting NEW onset only)
    panel_clean = panel[~panel["in_shortage"]].copy()

    panel_clean.to_parquet(out_path, index=False)
    print(f"\nPanel saved → {out_path}")
    print(f"  Rows (excl. in-shortage): {len(panel_clean):,}")
    print(f"\n  Base rates by split (y_6m):")
    for split in ["train", "val", "test"]:
        sp = panel_clean[panel_clean["split"] == split]
        if len(sp) == 0:
            continue
        rate = sp["y_6m"].mean() * 100
        pos = int(sp["y_6m"].sum())
        print(f"    {split:8s}: {len(sp):>8,} rows, {pos:>5,} positive ({rate:.2f}%)")

    return panel_clean
