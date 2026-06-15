"""
Phase 2 — Entity resolution: NDC labeler names ↔ FDA inspection/enforcement firm names.

Strategy (labeler-side-first):
1. Start from NDC labeler names in the drug universe (2,553 unique) — NOT from inspection firms.
2. For each labeler, fuzzy-match against manufacturing inspection firms
   (Drug Quality Assurance only — excludes Bioresearch Monitoring / clinical sites).
3. Output: drug_key → [fei_numbers] → inspection/citation/enforcement rows.

Also resolves: shortage.company_name → canonical labeler_name for label enrichment.
"""

import re
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process

from config import DATA_PROCESSED, MARKETING_CATEGORIES

CROSSWALK_PATH = Path(__file__).parent / "firm_crosswalk.csv"
OUT_FIRM_LINK = DATA_PROCESSED / "firm_link.parquet"
OUT_SHORTAGE_LINK = DATA_PROCESSED / "shortage_firm_link.parquet"
SHORTAGE_CROSSWALK_PATH = Path(__file__).parent / "shortage_crosswalk.csv"

# Project areas that indicate manufacturing (exclude Bioresearch Monitoring = clinical sites)
MFG_PROJECT_AREAS = {
    "Drug Quality Assurance",
    "Unapproved and Misbranded Drugs",
    "Over-the-Counter Drug Evaluation",
}

STRIP_SUFFIXES = re.compile(
    r"\b(INC|LLC|CORP|LTD|CO|COMPANY|PHARMACEUTICALS|PHARMACEUTICAL|PHARMA|PHARM|"
    r"LABORATORIES|LABORATORY|LABS|LAB|HOLDINGS|HOLDING|GROUP|USA|US|AMERICA|AMERICAS|"
    r"INTERNATIONAL|GLOBAL|AG|GMBH|PLC|SA|SAS|SRL|BV|NV|KK|INDUSTRIES|INDUSTRY|"
    r"HEALTHCARE|HEALTH|CARE|BIOSCIENCES|BIOSCIENCE|SCIENCES|SCIENCE|BIOLOGICS|"
    r"BIOSIMILARS|GENERICS|GENERIC|FORMERLY|DBA|FKA)\b\.?",
    re.IGNORECASE,
)


def normalize_firm(name: str) -> str:
    if not name or not str(name).strip():
        return ""
    s = str(name).upper()
    s = re.sub(r"[,\.\(\)\[\]\-\/\\&;]", " ", s)
    s = STRIP_SUFFIXES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_crosswalk() -> dict[str, str]:
    if not CROSSWALK_PATH.exists():
        return {}
    df = pd.read_csv(CROSSWALK_PATH, comment="#")
    result = {}
    for _, row in df.iterrows():
        k = str(row.get("fei_name_normalized", "")).strip().upper()
        v = str(row.get("canonical_name", "")).strip()
        if k and v:
            result[k] = v
    return result


def _fuzzy_match_one(query_norm: str, choices: list[str], threshold: int = 92) -> tuple[str, float] | None:
    """Return (best_match_norm, score) or None if no match above threshold."""
    if not query_norm or not choices:
        return None
    tokens = query_norm.split()

    # For single-token queries (e.g. "PFIZER", "NOVARTIS"):
    # find the best inspection firm whose normalized name STARTS WITH this token.
    # Require ≥5 chars to avoid "RJ"→"RJ GENERAL" type noise.
    if len(tokens) == 1 and len(tokens[0]) >= 5:
        token = tokens[0]
        pattern = re.compile(r"^" + re.escape(token) + r"\b")
        candidates = [c for c in choices if pattern.match(c)]
        if candidates:
            # Prefer the shortest match (most specific to that company)
            best = min(candidates, key=len)
            return best, 95.0
        return None

    # Multi-token: skip if max token is very short (risk of false positive)
    if max(len(t) for t in tokens) < 5:
        return None

    result = process.extractOne(
        query_norm, choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,
    )
    if result:
        return result[0], result[1]
    return None


def build_firm_link(force: bool = False) -> pd.DataFrame:
    """
    Resolve NDC labeler names → FDA inspection FEI numbers.
    Returns DataFrame: labeler_name, labeler_norm, fei_number, firm_name, match_method, match_score.
    """
    OUT_FIRM_LINK.parent.mkdir(parents=True, exist_ok=True)

    if OUT_FIRM_LINK.exists() and not force:
        print(f"Using cached firm_link: {OUT_FIRM_LINK}")
        return pd.read_parquet(OUT_FIRM_LINK)

    ndc = pd.read_parquet(DATA_PROCESSED / "ndc.parquet")
    inspections = pd.read_parquet(DATA_PROCESSED / "inspections.parquet")
    compliance = pd.read_parquet(DATA_PROCESSED / "compliance.parquet")
    enforcement = pd.read_parquet(DATA_PROCESSED / "enforcement.parquet")
    crosswalk = _load_crosswalk()

    # Drug universe labelers only
    rx_ndc = ndc[ndc["marketing_category"].isin(MARKETING_CATEGORIES)]
    labelers = rx_ndc["labeler_name"].dropna().unique().tolist()
    print(f"NDC labelers in drug universe: {len(labelers):,}")

    # Manufacturing inspections only (exclude clinical Bioresearch Monitoring)
    mfg_insp = inspections[
        inspections["project_area"].isin(MFG_PROJECT_AREAS)
    ].copy()
    print(f"Manufacturing inspection firm names: {mfg_insp['firm_name'].nunique():,}")

    # Build lookup: norm → (firm_name, fei_number)
    insp_firm_records = (
        mfg_insp[["firm_name", "fei_number"]]
        .drop_duplicates("firm_name")
        .copy()
    )
    insp_firm_records["firm_norm"] = insp_firm_records["firm_name"].apply(normalize_firm)
    insp_firm_records = insp_firm_records.drop_duplicates("firm_norm")
    norm_to_fei = insp_firm_records.set_index("firm_norm")[["firm_name", "fei_number"]].to_dict("index")
    insp_norms = list(norm_to_fei.keys())

    # Compliance firm lookup (no FEI number)
    comp_firms = compliance["firm_name"].dropna().unique().tolist()
    comp_norms_map = {normalize_firm(f): f for f in comp_firms}
    comp_norms = list(comp_norms_map.keys())

    # Enforcement recalling firms
    enf_firms = enforcement["recalling_firm"].dropna().unique().tolist()
    enf_norms_map = {normalize_firm(f): f for f in enf_firms}
    enf_norms = list(enf_norms_map.keys())

    print(f"Resolving {len(labelers):,} labelers → inspections/compliance/enforcement...")

    links = []
    matched_insp = 0
    matched_comp = 0
    matched_enf = 0

    for labeler in labelers:
        labeler_norm = normalize_firm(labeler)
        if not labeler_norm:
            continue

        # Crosswalk override
        if labeler_norm in crosswalk:
            target = crosswalk[labeler_norm]
            target_norm = normalize_firm(target)
            if target_norm in norm_to_fei:
                rec = norm_to_fei[target_norm]
                links.append({
                    "labeler_name": labeler, "labeler_norm": labeler_norm,
                    "fei_number": rec["fei_number"], "insp_firm_name": rec["firm_name"],
                    "match_method": "crosswalk", "match_score": 100,
                })
            matched_insp += 1
            continue

        # Exact match on inspections
        if labeler_norm in norm_to_fei:
            rec = norm_to_fei[labeler_norm]
            links.append({
                "labeler_name": labeler, "labeler_norm": labeler_norm,
                "fei_number": rec["fei_number"], "insp_firm_name": rec["firm_name"],
                "match_method": "exact", "match_score": 100,
            })
            matched_insp += 1
            continue

        # Fuzzy match on inspections
        hit = _fuzzy_match_one(labeler_norm, insp_norms, threshold=92)
        if hit:
            best_norm, score = hit
            rec = norm_to_fei[best_norm]
            links.append({
                "labeler_name": labeler, "labeler_norm": labeler_norm,
                "fei_number": rec["fei_number"], "insp_firm_name": rec["firm_name"],
                "match_method": "fuzzy", "match_score": score,
            })
            matched_insp += 1

    df = pd.DataFrame(links) if links else pd.DataFrame(columns=[
        "labeler_name", "labeler_norm", "fei_number", "insp_firm_name",
        "match_method", "match_score",
    ])

    df.to_parquet(OUT_FIRM_LINK, index=False)

    pct = 100 * matched_insp / len(labelers) if labelers else 0
    print(f"\nFirm resolution (labeler → inspection FEI):")
    print(f"  Labelers resolved: {matched_insp:,} / {len(labelers):,} ({pct:.1f}%)")
    print(f"  Link rows: {len(df):,}")
    if len(df) > 0:
        print(f"  Match methods: {df['match_method'].value_counts().to_dict()}")
    print(f"  Saved → {OUT_FIRM_LINK}")
    if pct < 40:
        print(f"  NOTE: {100-pct:.0f}% unresolved labelers have no manufacturing inspection record.")
        print(f"  This is expected for OTC, compounders, foreign-only, and discontinued firms.")

    return df


def _load_shortage_crosswalk() -> dict[str, tuple[str, str]]:
    """Returns {company_norm: (fei_number, insp_firm_name)} from manual crosswalk."""
    if not SHORTAGE_CROSSWALK_PATH.exists():
        return {}
    result = {}
    with open(SHORTAGE_CROSSWALK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            norm, fei, firm = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if norm and fei and fei != "fei_number":
                result[norm] = (fei, firm)
    return result


def build_shortage_firm_link(force: bool = False) -> pd.DataFrame:
    """
    Resolve shortage.company_name → FEI number via:
      1. Manual shortage_crosswalk.csv (highest priority)
      2. Exact match on NDC labeler_norm in firm_link
      3. Fuzzy match on NDC labeler_norm in firm_link
    Returns DataFrame: company_name, labeler_name, fei_number, match_method, match_score.
    """
    OUT_SHORTAGE_LINK.parent.mkdir(parents=True, exist_ok=True)

    if OUT_SHORTAGE_LINK.exists() and not force:
        print(f"Using cached shortage_firm_link: {OUT_SHORTAGE_LINK}")
        return pd.read_parquet(OUT_SHORTAGE_LINK)

    shorts = pd.read_parquet(DATA_PROCESSED / "shortages_raw.parquet")
    firm_link = build_firm_link(force=False)
    shortage_xwalk = _load_shortage_crosswalk()

    companies = shorts["company_name"].dropna().unique().tolist()
    companies = [c for c in companies if c]
    print(f"\nResolving {len(companies):,} shortage company names → labeler/FEI...")
    print(f"  Manual crosswalk entries: {len(shortage_xwalk)}")

    labeler_norms = firm_link["labeler_norm"].unique().tolist() if not firm_link.empty else []
    norm_to_labeler = (
        firm_link.drop_duplicates("labeler_norm")
        .set_index("labeler_norm")[["labeler_name", "fei_number"]]
        .to_dict("index")
    ) if not firm_link.empty else {}

    links = []
    matched = 0
    for company in companies:
        company_norm = normalize_firm(company)
        if not company_norm:
            continue

        # 1. Manual crosswalk (parent manufacturer override)
        if company_norm in shortage_xwalk:
            fei, firm = shortage_xwalk[company_norm]
            links.append({
                "company_name": company, "labeler_name": firm,
                "fei_number": fei,
                "match_method": "crosswalk", "match_score": 100,
            })
            matched += 1
            continue

        # 2. Exact match on labeler_norm
        if company_norm in norm_to_labeler:
            rec = norm_to_labeler[company_norm]
            links.append({
                "company_name": company, "labeler_name": rec["labeler_name"],
                "fei_number": rec["fei_number"],
                "match_method": "exact", "match_score": 100,
            })
            matched += 1
            continue

        # 3. Fuzzy match
        hit = _fuzzy_match_one(company_norm, labeler_norms, threshold=90)
        if hit:
            best_norm, score = hit
            rec = norm_to_labeler[best_norm]
            links.append({
                "company_name": company, "labeler_name": rec["labeler_name"],
                "fei_number": rec["fei_number"],
                "match_method": "fuzzy", "match_score": score,
            })
            matched += 1

    df = pd.DataFrame(links) if links else pd.DataFrame(columns=[
        "company_name", "labeler_name", "fei_number", "match_method", "match_score",
    ])

    df.to_parquet(OUT_SHORTAGE_LINK, index=False)
    pct = 100 * matched / len(companies) if companies else 0
    print(f"  Shortage companies resolved: {matched:,} / {len(companies):,} ({pct:.1f}%)")
    if len(df) > 0:
        print(f"  Methods: {df['match_method'].value_counts().to_dict()}")
    print(f"  Saved → {OUT_SHORTAGE_LINK}")

    return df
