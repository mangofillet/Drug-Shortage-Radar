"""
Phase 1d — FDA compliance actions (warning letters, injunctions, seizures).

Primary: FDA Data Dashboard OII API  POST /v1/compliance_actions
  Set env vars: FDA_DASHBOARD_USER (email), FDA_DASHBOARD_KEY (api key)
  Register at: https://datadashboard.fda.gov/oii/api/index.htm

Fallback: empty DataFrame with correct schema — pipeline continues without this source,
noting reduced manufacturing-risk signal coverage as a documented limitation.
"""

import os
import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_RAW, DATA_PROCESSED

OII_BASE = "https://api-datadashboard.fda.gov/v1"
RAW_PATH = DATA_RAW / "compliance_raw.parquet"
OUT_PATH = DATA_PROCESSED / "compliance.parquet"

# Columns verified against live API responses — no date field exists on this endpoint
COMPLIANCE_COLUMNS = [
    "FEINumber", "LegalName", "ActionType", "City", "State", "ProductType", "Region",
]
PAGE_SIZE = 5000


def _oii_headers() -> dict | None:
    user = os.environ.get("FDA_DASHBOARD_USER", "")
    key = os.environ.get("FDA_DASHBOARD_KEY", "")
    if not user or not key:
        return None
    return {
        "Content-Type": "application/json",
        "Authorization-User": user,
        "Authorization-Key": key,
    }


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=60))
def _oii_post(endpoint: str, headers: dict, body: dict) -> requests.Response:
    r = requests.post(f"{OII_BASE}/{endpoint}", json=body, headers=headers, timeout=120)
    r.raise_for_status()
    return r


def _fetch_all_compliance() -> list[dict]:
    """
    Fetch all compliance_actions. The endpoint has no date column and no pagination support —
    one request returns the full dataset (2,600–3,000 rows as of 2026).
    Shard by ActionType to stay within the 5000-row cap.
    """
    headers = _oii_headers()
    if headers is None:
        return []

    all_rows: list[dict] = []
    for action_type in ("Warning Letter", "Injunction", "Seizure", "Import Alert", "Consent Decree"):
        for sort_dir in ("asc", "desc"):
            body = {
                "sort": "FEINumber",
                "sortorder": sort_dir,
                "filters": {"ActionType": [action_type], "ProductType": ["Drugs"]},
                "columns": COMPLIANCE_COLUMNS,
            }
            try:
                r = _oii_post("compliance_actions", headers, body)
                data = r.json()
                rows = data.get("data", data.get("results", [])) if isinstance(data, dict) else data
                all_rows.extend(rows)
                print(f"    {action_type}/{sort_dir}: {len(rows)} rows")
                time.sleep(0.3)
            except Exception as e:
                print(f"  WARN compliance_actions {action_type}/{sort_dir}: {e}")

    # Deduplicate (asc+desc may overlap)
    seen = set()
    unique = []
    for r in all_rows:
        key = (r.get("FEINumber", ""), r.get("LegalName", ""), r.get("ActionType", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def ingest_compliance(force: bool = False) -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    _empty = pd.DataFrame(columns=[
        "fei_number", "firm_name", "action_type", "city", "state", "product_type", "region",
    ])

    if not _oii_headers():
        print("  SKIP compliance: FDA_DASHBOARD_USER / FDA_DASHBOARD_KEY not set.")
        print("  Register at: https://datadashboard.fda.gov/oii/api/index.htm")
        if not OUT_PATH.exists():
            _empty.to_parquet(OUT_PATH, index=False)
        return _empty

    if RAW_PATH.exists() and not force:
        print(f"Using cached compliance data: {RAW_PATH}")
        df_raw = pd.read_parquet(RAW_PATH)
    else:
        print("Fetching compliance_actions via OII API...")
        rows = _fetch_all_compliance()
        df_raw = pd.DataFrame(rows)
        df_raw.to_parquet(RAW_PATH, index=False)
        print(f"  Raw compliance: {len(df_raw):,} rows")

    if df_raw.empty:
        _empty.to_parquet(OUT_PATH, index=False)
        return _empty

    rename = {
        "FEINumber": "fei_number",
        "LegalName": "firm_name",
        "ActionType": "action_type",
        "City": "city",
        "State": "state",
        "ProductType": "product_type",
        "Region": "region",
    }
    df = df_raw.rename(columns={k: v for k, v in rename.items() if k in df_raw.columns})

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} compliance rows → {OUT_PATH}")
    return df
