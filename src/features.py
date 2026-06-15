"""
Phase 4 — Feature engineering (leakage-safe).

All features computed as-of end of panel month t.
Vectorized: no row-by-row loops over the panel.
"""

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, MARKETING_CATEGORIES


def _parse_route(route_str) -> str:
    if not route_str or pd.isna(route_str):
        return ""
    return str(route_str).strip("[]'\" ").split(",")[0].strip("'\" ").lower()


def _load_sources() -> dict:
    return {
        "ndc": pd.read_parquet(DATA_PROCESSED / "ndc.parquet"),
        "inspections": pd.read_parquet(DATA_PROCESSED / "inspections.parquet"),
        "citations": pd.read_parquet(DATA_PROCESSED / "citations.parquet"),
        "compliance": pd.read_parquet(DATA_PROCESSED / "compliance.parquet"),
        "enforcement": pd.read_parquet(DATA_PROCESSED / "enforcement.parquet"),
        "nadac": pd.read_parquet(DATA_PROCESSED / "nadac.parquet"),
        "shortages": pd.read_parquet(DATA_PROCESSED / "shortages_raw.parquet"),
        "firm_link": pd.read_parquet(DATA_PROCESSED / "firm_link.parquet"),
    }


def _prep_ndc(ndc: pd.DataFrame) -> pd.DataFrame:
    rx = ndc[ndc["marketing_category"].isin(MARKETING_CATEGORIES)].copy()
    rx["route_norm"] = rx["route"].apply(_parse_route)
    rx["generic_norm"] = rx["generic_name"].str.lower().str.strip().fillna("")
    rx["drug_key"] = rx["generic_norm"] + "|" + rx["route_norm"]
    rx["marketing_start_date"] = pd.to_datetime(rx["marketing_start_date"], errors="coerce")
    rx["marketing_end_date"] = pd.to_datetime(rx["marketing_end_date"], errors="coerce")
    rx["is_anda"] = rx["marketing_category"] == "ANDA"
    rx["route_injectable"] = rx["route_norm"].str.contains(
        r"inj|intravenous|\biv\b", case=False, na=False, regex=True
    )
    rx["dea_scheduled"] = rx["dea_schedule"].notna() & (rx["dea_schedule"] != "")
    return rx


def _drug_fei_map(ndc_rx: pd.DataFrame, firm_link: pd.DataFrame) -> pd.DataFrame:
    """Return (drug_key, fei_number) by joining NDC labeler_name → firm_link."""
    labeler_fei = firm_link[["labeler_name", "fei_number"]].drop_duplicates()
    drug_labeler = ndc_rx[["drug_key", "labeler_name"]].drop_duplicates()
    return drug_labeler.merge(labeler_fei, on="labeler_name")[["drug_key", "fei_number"]].drop_duplicates()


def _windowed_insp_features(panel: pd.DataFrame, drug_fei: pd.DataFrame, inspections: pd.DataFrame) -> pd.DataFrame:
    """Vectorized inspection count features by time window."""
    insp = inspections[inspections["inspection_end_date"].notna()].copy()
    insp["inspection_end_date"] = pd.to_datetime(insp["inspection_end_date"])

    # drug_key → fei_number → inspections
    drug_insp = drug_fei.merge(
        insp[["fei_number", "inspection_end_date", "classification"]].drop_duplicates(),
        on="fei_number"
    )
    if drug_insp.empty:
        out = panel[["drug_key", "month"]].assign(
            n_oai_12m=0, n_oai_36m=0, n_vai_12m=0, n_insp_12m=0, n_insp_24m=0,
            worst_class_24m=0,
        )
        out["months_since_last_insp"] = np.nan
        return out

    # Cross-join panel with drug_insp on drug_key, filter by time window
    merged = panel[["drug_key", "month"]].merge(drug_insp, on="drug_key", how="left")
    merged["t_end"] = merged["month"] + pd.offsets.MonthEnd(0)
    merged["d"] = merged["inspection_end_date"]

    def _flag(df, months):
        start = df["t_end"] - pd.DateOffset(months=months)
        return (df["d"] >= start) & (df["d"] <= df["t_end"])

    merged["in_12m"] = _flag(merged, 12)
    merged["in_24m"] = _flag(merged, 24)
    merged["in_36m"] = _flag(merged, 36)

    merged["in_24m_any"] = _flag(merged, 24)
    cl = merged["classification"]
    merged["oai_12m"] = (cl == "OAI") & merged["in_12m"]
    merged["oai_36m"] = (cl == "OAI") & merged["in_36m"]
    merged["vai_12m"] = (cl == "VAI") & merged["in_12m"]
    merged["any_12m"] = merged["in_12m"]
    merged["any_24m"] = merged["in_24m_any"]
    # Worst classification in 24m: OAI=2, VAI=1, NAI=0
    cl_num = cl.map({"OAI": 2, "VAI": 1, "NAI": 0}).fillna(-1)
    merged["worst_24m_num"] = cl_num.where(merged["in_24m"], -1)
    # Most recent past inspection date (<= t_end) for recency feature
    merged["past_insp_date"] = merged["d"].where(merged["d"] <= merged["t_end"])

    agg = merged.groupby(["drug_key", "month"], sort=False).agg(
        n_oai_12m=("oai_12m", "sum"),
        n_oai_36m=("oai_36m", "sum"),
        n_vai_12m=("vai_12m", "sum"),
        n_insp_12m=("any_12m", "sum"),
        n_insp_24m=("any_24m", "sum"),
        worst_class_24m=("worst_24m_num", "max"),
        last_insp_date=("past_insp_date", "max"),
        t_end=("t_end", "first"),
    ).reset_index()

    agg["worst_class_24m"] = agg["worst_class_24m"].clip(lower=0).astype(int)
    agg["months_since_last_insp"] = (
        (agg["t_end"] - agg["last_insp_date"]).dt.days / 30.44
    )
    agg.drop(columns=["last_insp_date", "t_end"], inplace=True)
    return agg


def _windowed_recall_features(panel: pd.DataFrame, enforcement: pd.DataFrame) -> pd.DataFrame:
    """Recalls matched by first ~12 chars of generic name."""
    enf = enforcement.copy()
    enf["recall_initiation_date"] = pd.to_datetime(enf["recall_initiation_date"], errors="coerce")
    enf = enf[enf["recall_initiation_date"].notna()]

    # Extract first word of product description as rough drug name match
    enf["name_tok"] = (
        enf["product_description"].str.lower()
        .str.extract(r"^([a-z]{4,})", expand=False)
        .fillna("")
    )
    panel["name_tok"] = panel["drug_key"].str.split("|").str[0].str.split().str[0]

    merged = panel[["drug_key", "month", "name_tok"]].merge(
        enf[["name_tok", "recall_initiation_date", "recall_class"]],
        on="name_tok", how="left"
    )
    merged["t_end"] = merged["month"] + pd.offsets.MonthEnd(0)
    merged["d"] = merged["recall_initiation_date"]
    merged["in_6m"] = (merged["d"] >= merged["t_end"] - pd.DateOffset(months=6)) & (merged["d"] <= merged["t_end"])
    merged["in_12m"] = (merged["d"] >= merged["t_end"] - pd.DateOffset(months=12)) & (merged["d"] <= merged["t_end"])
    merged["in_24m"] = (merged["d"] >= merged["t_end"] - pd.DateOffset(months=24)) & (merged["d"] <= merged["t_end"])
    merged["class1_24m"] = (merged["recall_class"] == "Class I") & merged["in_24m"]
    merged["past_recall_date"] = merged["d"].where(merged["d"] <= merged["t_end"])

    agg = merged.groupby(["drug_key", "month"], sort=False).agg(
        recalls_6m=("in_6m", "sum"),
        recalls_12m=("in_12m", "sum"),
        recalls_class1_24m=("class1_24m", "sum"),
        last_recall_date=("past_recall_date", "max"),
        t_end=("t_end", "first"),
    ).reset_index()
    agg["months_since_last_recall"] = (
        (agg["t_end"] - agg["last_recall_date"]).dt.days / 30.44
    )
    agg.drop(columns=["last_recall_date", "t_end"], inplace=True)

    panel.drop(columns=["name_tok"], inplace=True)
    return agg


def _market_structure_features(panel: pd.DataFrame, ndc_rx: pd.DataFrame) -> pd.DataFrame:
    """Active labelers, HHI, injectable flag, DEA schedule, drug age."""
    # Build (drug_key, labeler_name, marketing_start, marketing_end) table
    labeler_mkt = ndc_rx[[
        "drug_key", "labeler_name", "marketing_start_date", "marketing_end_date",
        "is_anda", "route_injectable", "dea_scheduled"
    ]].drop_duplicates(["drug_key", "labeler_name"])

    # Drug-level static flags (time-invariant approximation)
    drug_flags = (
        ndc_rx.groupby("drug_key")
        .agg(
            route_injectable=("route_injectable", "any"),
            dea_scheduled=("dea_scheduled", "any"),
            first_marketed=("marketing_start_date", "min"),
        )
        .reset_index()
    )
    drug_flags["first_marketed"] = pd.to_datetime(drug_flags["first_marketed"])

    # Cross-join panel with labeler_mkt, filter to active labelers at time t
    merged = panel[["drug_key", "month"]].merge(labeler_mkt, on="drug_key", how="left")
    merged["t_end"] = merged["month"] + pd.offsets.MonthEnd(0)
    active = (
        (merged["marketing_start_date"].isna() | (merged["marketing_start_date"] <= merged["t_end"])) &
        (merged["marketing_end_date"].isna() | (merged["marketing_end_date"] > merged["month"]))
    )
    merged["active"] = active
    merged["active_anda"] = active & merged["is_anda"]

    agg = merged.groupby(["drug_key", "month"], sort=False).agg(
        n_active_labelers=("active", "sum"),
        n_active_anda=("active_anda", "sum"),
    ).reset_index()
    agg["n_active_labelers"] = agg["n_active_labelers"].clip(lower=0)
    agg["is_generic"] = (agg["n_active_anda"] > 0).astype(int)
    # HHI = 1/n under equal market share assumption
    agg["labeler_hhi"] = np.where(agg["n_active_labelers"] > 0, 1.0 / agg["n_active_labelers"], 1.0)

    # Join drug-level static flags
    agg = agg.merge(drug_flags[["drug_key", "route_injectable", "dea_scheduled", "first_marketed"]], on="drug_key", how="left")
    agg["route_injectable"] = agg["route_injectable"].fillna(False).astype(int)
    agg["dea_scheduled"] = agg["dea_scheduled"].fillna(False).astype(int)
    agg["drug_age_years"] = (agg["month"] - agg["first_marketed"]).dt.days / 365.25
    agg.drop(columns=["first_marketed"], inplace=True)

    return agg


def _supplier_exit_features(panel: pd.DataFrame, ndc_rx: pd.DataFrame) -> pd.DataFrame:
    """
    Supplier-exit dynamics — a manufacturer discontinuing a product is often the
    proximate trigger of a shortage. Leakage-safe: a labeler's marketing_end_date is
    only counted once it is on/before month t (a future-dated end relative to t is
    excluded), and marketing_start_date likewise.

    Features:
      labelers_exited_12m     — distinct labelers whose product ended in (t-12m, t]
      labelers_entered_12m    — distinct labelers who started in (t-12m, t]
      net_labeler_change_12m  — entrants minus exits (negative = thinning supply base)
      months_since_last_exit  — recency of the most recent exit (NaN if none observed)
    """
    labeler_mkt = ndc_rx[[
        "drug_key", "labeler_name", "marketing_start_date", "marketing_end_date"
    ]].drop_duplicates(["drug_key", "labeler_name"])

    merged = panel[["drug_key", "month"]].merge(labeler_mkt, on="drug_key", how="left")
    merged["t_end"] = merged["month"] + pd.offsets.MonthEnd(0)
    end = merged["marketing_end_date"]
    start = merged["marketing_start_date"]
    window_start = merged["t_end"] - pd.DateOffset(months=12)

    merged["exit_12m"] = (end > window_start) & (end <= merged["t_end"])
    merged["entrant_12m"] = (start > window_start) & (start <= merged["t_end"])
    merged["past_exit_date"] = end.where(end <= merged["t_end"])

    agg = merged.groupby(["drug_key", "month"], sort=False).agg(
        labelers_exited_12m=("exit_12m", "sum"),
        labelers_entered_12m=("entrant_12m", "sum"),
        last_exit_date=("past_exit_date", "max"),
        t_end=("t_end", "first"),
    ).reset_index()

    agg["labelers_exited_12m"] = agg["labelers_exited_12m"].fillna(0).astype(int)
    agg["labelers_entered_12m"] = agg["labelers_entered_12m"].fillna(0).astype(int)
    agg["net_labeler_change_12m"] = agg["labelers_entered_12m"] - agg["labelers_exited_12m"]
    agg["months_since_last_exit"] = (
        (agg["t_end"] - agg["last_exit_date"]).dt.days / 30.44
    )
    agg.drop(columns=["last_exit_date", "t_end"], inplace=True)
    return agg


def _nadac_features(panel: pd.DataFrame, ndc_rx: pd.DataFrame, nadac: pd.DataFrame) -> pd.DataFrame:
    """NADAC price level and trend via drug_name text prefix match."""
    if nadac.empty or "nadac_per_unit" not in nadac.columns:
        return panel[["drug_key", "month"]].assign(
            nadac_latest_price=np.nan, nadac_price_trend_12m=np.nan
        )

    nadac = nadac.copy()
    nadac["effective_date"] = pd.to_datetime(nadac["effective_date"], errors="coerce")
    nadac = nadac[nadac["effective_date"].notna() & nadac["nadac_per_unit"].notna()]

    # Match NADAC drug_name to drug_key by first word of generic name
    nadac["name_tok"] = nadac["drug_name"].str.lower().str.split().str[0].fillna("")
    panel_copy = panel[["drug_key", "month"]].copy()
    panel_copy["name_tok"] = panel_copy["drug_key"].str.split("|").str[0].str.split().str[0]

    merged = panel_copy.merge(
        nadac[["name_tok", "effective_date", "nadac_per_unit"]],
        on="name_tok", how="left"
    )
    merged["t_end"] = merged["month"] + pd.offsets.MonthEnd(0)
    in_window = (
        (merged["effective_date"] <= merged["t_end"]) &
        (merged["effective_date"] >= merged["t_end"] - pd.DateOffset(months=12))
    )
    merged_w = merged[in_window].copy()

    # Latest price = most recent per (drug_key, month)
    latest = (
        merged_w.sort_values("effective_date")
        .groupby(["drug_key", "month"])["nadac_per_unit"]
        .last()
        .reset_index()
        .rename(columns={"nadac_per_unit": "nadac_latest_price"})
    )

    # Price trend: slope of log(price) over 12m window
    def _slope(prices):
        if len(prices) < 3:
            return np.nan
        lp = np.log(np.clip(prices.values, 1e-6, None))
        x = np.arange(len(lp))
        return float(np.polyfit(x, lp, 1)[0])

    trend = (
        merged_w.sort_values("effective_date")
        .groupby(["drug_key", "month"])["nadac_per_unit"]
        .apply(_slope)
        .reset_index()
        .rename(columns={"nadac_per_unit": "nadac_price_trend_12m"})
    )

    result = panel[["drug_key", "month"]].merge(latest, on=["drug_key", "month"], how="left")
    result = result.merge(trend, on=["drug_key", "month"], how="left")
    return result


import re as _re
_DOSAGE_FORM_RE = _re.compile(
    r"[\s,]+(injection|injectable|infusion|intravenous|iv\b|oral\b|tablet|tablets|capsule|capsules|"
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
_PAREN_RE = _re.compile(r"\s*\(.*?\)")


def _strip_dosage_form(name: str) -> str:
    s = _PAREN_RE.sub("", str(name)).strip()
    return _DOSAGE_FORM_RE.sub("", s).strip().rstrip(",").strip().lower()


def _shortage_history_features(panel: pd.DataFrame, shortages: pd.DataFrame) -> pd.DataFrame:
    """Past shortage count and months since last shortage."""
    shortages = shortages[shortages["onset_date"].notna()].copy()
    shortages["onset_date"] = pd.to_datetime(shortages["onset_date"])
    # Strip dosage form so "atropine sulfate injection" → "atropine sulfate" matches drug_key
    shortages["generic_norm"] = shortages["generic_name"].apply(_strip_dosage_form)
    shortages["onset_month"] = shortages["onset_date"].dt.to_period("M").dt.to_timestamp()

    panel_copy = panel[["drug_key", "month"]].copy()
    panel_copy["generic_norm"] = panel_copy["drug_key"].str.split("|").str[0]

    merged = panel_copy.merge(
        shortages[["generic_norm", "onset_month"]].drop_duplicates(),
        on="generic_norm", how="left"
    )
    # Only past onsets (strictly before panel month)
    past = merged[merged["onset_month"] < merged["month"]].copy()

    counts = (
        past.groupby(["drug_key", "month"])
        .agg(
            past_shortage_count=("onset_month", "count"),
            last_onset=("onset_month", "max"),
        )
        .reset_index()
    )
    counts["months_since_last_shortage"] = (
        (counts["month"] - counts["last_onset"]).dt.days / 30.44
    )
    counts.drop(columns=["last_onset"], inplace=True)

    result = panel[["drug_key", "month"]].merge(counts, on=["drug_key", "month"], how="left")
    result["past_shortage_count"] = result["past_shortage_count"].fillna(0).astype(int)
    return result


def _related_shortage_features(panel: pd.DataFrame, shortages: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-form supply-stress signal: at month t, how many OTHER presentations
    of the same molecule (generic_norm) are in an *active* shortage right now?

    Leakage-safe: a shortage is "active and knowable at t" iff onset_month <= t
    and it has not yet resolved (end_month >= t). Onset is past; non-resolution
    is observable in real time. The drug's own active-shortage months were already
    removed from the panel, so this measures sibling/competitor stress on the
    same molecule — a leading indicator of the drug's own onset.
    """
    s = shortages[shortages["onset_date"].notna()].copy()
    s["onset_month"] = pd.to_datetime(s["onset_date"]).dt.to_period("M").dt.to_timestamp()
    # End of active window: explicit resolution if known, else last time observed current
    end = pd.to_datetime(s["resolved_date_approx"], errors="coerce")
    last_seen = pd.to_datetime(s["last_seen_date"], errors="coerce")
    s["end_month"] = end.fillna(last_seen).dt.to_period("M").dt.to_timestamp()
    s["generic_norm"] = s["generic_name"].apply(_strip_dosage_form)
    s = s.dropna(subset=["generic_norm", "onset_month", "end_month"])
    s = s[s["generic_norm"] != ""]
    # Each (generic, presentation, company) is one supply line
    s["line_id"] = (s["presentation_norm"].fillna("") + "|" + s["company_norm"].fillna(""))
    intervals = s[["generic_norm", "onset_month", "end_month", "line_id"]].drop_duplicates()

    # Unique generic-month grid from panel (keeps the merge small)
    grid = panel[["drug_key", "month"]].copy()
    grid["generic_norm"] = grid["drug_key"].str.split("|").str[0]
    gm = grid[["generic_norm", "month"]].drop_duplicates()

    merged = gm.merge(intervals, on="generic_norm", how="inner")
    active = (merged["onset_month"] <= merged["month"]) & (merged["end_month"] >= merged["month"])
    merged = merged[active]
    counts = (
        merged.groupby(["generic_norm", "month"])["line_id"]
        .nunique()
        .reset_index(name="related_shortages_active")
    )

    result = grid.merge(counts, on=["generic_norm", "month"], how="left")
    result["related_shortages_active"] = result["related_shortages_active"].fillna(0).astype(int)
    return result[["drug_key", "month", "related_shortages_active"]]


def build_features(panel: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    out_path = DATA_PROCESSED / "features.parquet"
    if out_path.exists() and not force:
        print(f"Using cached features: {out_path}")
        return pd.read_parquet(out_path)

    src = _load_sources()
    ndc_rx = _prep_ndc(src["ndc"])
    drug_fei = _drug_fei_map(ndc_rx, src["firm_link"])

    print(f"Building features for {len(panel):,} panel rows...")
    print(f"  drug_key→FEI links: {len(drug_fei):,}")

    print("  [1/5] Market structure features...")
    mkt = _market_structure_features(panel, ndc_rx)

    print("  [2/5] Inspection features...")
    insp_feat = _windowed_insp_features(panel, drug_fei, src["inspections"])

    print("  [3/5] Recall features...")
    recall_feat = _windowed_recall_features(panel, src["enforcement"])

    print("  [4/7] NADAC price features...")
    nadac_feat = _nadac_features(panel, ndc_rx, src["nadac"])

    print("  [5/7] Supplier-exit dynamics...")
    exit_feat = _supplier_exit_features(panel, ndc_rx)

    print("  [6/7] Shortage history features...")
    hist_feat = _shortage_history_features(panel, src["shortages"])

    print("  [7/7] Related-form shortage spillover...")
    related_feat = _related_shortage_features(panel, src["shortages"])

    # Merge all feature blocks onto panel
    features = panel[["drug_key", "month", "y_6m", "y_3m", "split"]].copy()
    for feat_df in [mkt, insp_feat, recall_feat, nadac_feat, exit_feat, hist_feat, related_feat]:
        features = features.merge(feat_df, on=["drug_key", "month"], how="left")

    # Compliance: no date column — binary "ever cited" flag
    ever_cited_firms = set(src["compliance"]["firm_name"].dropna().unique())
    drug_labeler = ndc_rx[["drug_key", "labeler_name"]].drop_duplicates()
    fl = src["firm_link"][["labeler_name", "insp_firm_name"]].drop_duplicates()
    drug_insp_firm = drug_labeler.merge(fl, on="labeler_name")[["drug_key", "insp_firm_name"]]
    ever_cited = (
        drug_insp_firm[drug_insp_firm["insp_firm_name"].isin(ever_cited_firms)]
        .groupby("drug_key")
        .size()
        .gt(0)
        .reset_index()
        .rename(columns={0: "firm_ever_compliance_action"})
    )
    features = features.merge(ever_cited, on="drug_key", how="left")
    features["firm_ever_compliance_action"] = (
        features["firm_ever_compliance_action"].fillna(False).astype(int)
    )

    # Citation count for linked firms (time-windowed)
    citations = src["citations"].copy()
    citations["inspection_date"] = pd.to_datetime(citations["inspection_date"], errors="coerce")
    citations = citations[citations["inspection_date"].notna()]
    drug_cite = drug_fei.merge(
        citations[["fei_number", "inspection_date"]],
        on="fei_number", how="inner"
    )
    if not drug_cite.empty:
        cite_merged = panel[["drug_key", "month"]].merge(drug_cite, on="drug_key", how="left")
        cite_merged["t_end"] = cite_merged["month"] + pd.offsets.MonthEnd(0)
        in_24m = (
            (cite_merged["inspection_date"] >= cite_merged["t_end"] - pd.DateOffset(months=24)) &
            (cite_merged["inspection_date"] <= cite_merged["t_end"])
        )
        cite_counts = (
            cite_merged[in_24m]
            .groupby(["drug_key", "month"])
            .size()
            .reset_index(name="n_citations_24m")
        )
        features = features.merge(cite_counts, on=["drug_key", "month"], how="left")
    else:
        features["n_citations_24m"] = 0
    features["n_citations_24m"] = features["n_citations_24m"].fillna(0).astype(int)

    # Fill remaining count nulls (recency columns intentionally left NaN for LightGBM)
    insp_cols = ["n_oai_12m", "n_oai_36m", "n_vai_12m", "n_insp_12m", "n_insp_24m", "worst_class_24m"]
    recall_cols = ["recalls_6m", "recalls_12m", "recalls_class1_24m"]
    for col in insp_cols + recall_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    features.to_parquet(out_path, index=False)
    print(f"\nFeatures saved → {out_path}")
    print(f"  Shape: {features.shape}")
    null_rates = features.isnull().mean().sort_values(ascending=False).head(10)
    print(f"  Top null rates:\n{null_rates.to_string()}")
    print(f"\n  Feature columns: {[c for c in features.columns if c not in ('drug_key','month','y_6m','y_3m','split')]}")

    return features
