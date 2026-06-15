"""
Phase 1c — FDA inspections classifications + citations.

Primary: FDA Data Dashboard OII API (POST, Authorization-User + Authorization-Key headers).
  Endpoints: /v1/inspections_classifications, /v1/inspections_citations
  Key request: https://datadashboard.fda.gov/oii/api/index.htm
  Set env vars: FDA_DASHBOARD_USER (email), FDA_DASHBOARD_KEY (api key)

Fallback: returns empty DataFrames with correct schema so the pipeline still runs.
"""

import os
import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_RAW, DATA_PROCESSED

OII_BASE = "https://api-datadashboard.fda.gov/v1"
RAW_INSP_PATH = DATA_RAW / "inspections_raw.parquet"
OUT_INSP_PATH = DATA_PROCESSED / "inspections.parquet"
OUT_CITES_PATH = DATA_PROCESSED / "citations.parquet"

# Column names as returned by the FDA OII API (verified against live responses)
INSP_COLUMNS = [
    "FEINumber", "LegalName", "InspectionEndDate", "ClassificationCode",
    "ProjectArea", "City", "State", "InspectionID",
]
CITE_COLUMNS = [
    "FEINumber", "InspectionEndDate", "Citation", "ShortDescription",
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


def _fetch_all(endpoint: str, columns: list[str], filters: dict | None = None) -> list[dict]:
    headers = _oii_headers()
    if headers is None:
        return []

    all_rows: list[dict] = []
    page = 1
    while True:
        body = {
            "sort": columns[0],
            "sortorder": "asc",
            "filters": filters or {},
            "columns": columns,
            "page": page,
            "pageSize": PAGE_SIZE,
        }
        try:
            r = _oii_post(endpoint, headers, body)
            data = r.json()
            # API returns a list or {"data": [...], "totalCount": N}
            if isinstance(data, list):
                rows = data
                total = len(data)
            elif isinstance(data, dict):
                rows = data.get("data", data.get("results", []))
                total = data.get("totalCount", data.get("total", len(rows)))
            else:
                break

            if not rows:
                break
            all_rows.extend(rows)
            print(f"    page {page}: {len(rows)} rows (total so far: {len(all_rows):,} / {total:,})")

            if len(all_rows) >= total or len(rows) < PAGE_SIZE:
                break
            page += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  WARN {endpoint} page {page}: {e}")
            break

    return all_rows


def ingest_inspections(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_INSP_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_INSP_PATH.parent.mkdir(parents=True, exist_ok=True)

    _empty_insp = pd.DataFrame(columns=[
        "fei_number", "firm_name", "firm_address", "country_area",
        "inspection_end_date", "classification", "project_area",
    ])
    _empty_cites = pd.DataFrame(columns=[
        "fei_number", "inspection_date", "cfr_citation", "citation_description",
    ])

    if not _oii_headers():
        print("  SKIP inspections: FDA_DASHBOARD_USER / FDA_DASHBOARD_KEY not set.")
        print("  Register at: https://datadashboard.fda.gov/oii/api/index.htm")
        print("  Then: export FDA_DASHBOARD_USER=your@email.com FDA_DASHBOARD_KEY=yourkey")
        if not OUT_INSP_PATH.exists():
            _empty_insp.to_parquet(OUT_INSP_PATH, index=False)
            _empty_cites.to_parquet(OUT_CITES_PATH, index=False)
        return _empty_insp, _empty_cites

    if RAW_INSP_PATH.exists() and not force:
        print(f"Using cached inspections: {RAW_INSP_PATH}")
        df_raw = pd.read_parquet(RAW_INSP_PATH)
    else:
        print("Fetching FDA inspections_classifications via OII API...")
        # Filter to Drug program area to keep data manageable
        rows = _fetch_all(
            "inspections_classifications",
            INSP_COLUMNS,
            filters={"ProgramAreaDescription": "Drug"},
        )
        if not rows:
            # Try without filter — some versions use a different field name
            print("  Retrying without program-area filter...")
            rows = _fetch_all("inspections_classifications", INSP_COLUMNS)

        df_raw = pd.DataFrame(rows)
        df_raw.to_parquet(RAW_INSP_PATH, index=False)
        print(f"  Raw inspections: {len(df_raw):,} rows")

    # Normalize column names to pipeline schema
    rename = {
        "FEINumber": "fei_number",
        "LegalName": "firm_name",
        "InspectionEndDate": "inspection_end_date",
        "ClassificationCode": "classification",
        "ProjectArea": "project_area",
        "City": "city",
        "State": "state",
        "InspectionID": "inspection_id",
    }
    df_insp = df_raw.rename(columns={k: v for k, v in rename.items() if k in df_raw.columns})

    if "project_area" in df_insp.columns:
        mask = df_insp["project_area"].str.contains("drug|pharma", case=False, na=False)
        df_insp = df_insp[mask].copy()

    if "inspection_end_date" in df_insp.columns:
        df_insp["inspection_end_date"] = pd.to_datetime(
            df_insp["inspection_end_date"], errors="coerce"
        )

    df_insp.to_parquet(OUT_INSP_PATH, index=False)
    print(f"Saved {len(df_insp):,} drug inspection rows → {OUT_INSP_PATH}")

    # Citations
    print("Fetching inspections_citations via OII API...")
    cite_rows = _fetch_all("inspections_citations", CITE_COLUMNS)
    df_cites = pd.DataFrame(cite_rows)
    if not df_cites.empty:
        cite_rename = {
            "FEINumber": "fei_number",
            "InspectionEndDate": "inspection_date",
            "Citation": "cfr_citation",
            "ShortDescription": "citation_description",
        }
        df_cites = df_cites.rename(columns={k: v for k, v in cite_rename.items() if k in df_cites.columns})
        if "inspection_date" in df_cites.columns:
            df_cites["inspection_date"] = pd.to_datetime(df_cites["inspection_date"], errors="coerce")
    else:
        df_cites = _empty_cites

    df_cites.to_parquet(OUT_CITES_PATH, index=False)
    print(f"Saved {len(df_cites):,} citation rows → {OUT_CITES_PATH}")

    return df_insp, df_cites
