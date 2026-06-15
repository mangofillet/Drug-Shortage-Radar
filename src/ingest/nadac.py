"""
Phase 1f — CMS NADAC drug acquisition cost (data.medicaid.gov DKAN API).

Strategy: sample one effective_date per month per year-dataset (via DKAN conditions filter).
This gives one price snapshot per NDC per month, sufficient for price-level and price-trend features.
Full table download (~1.5M rows/year × 9 years) is impractical via the 5000-row paged API.
"""

import time
import io

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_RAW, DATA_PROCESSED

# Per-year dataset IDs. Each spans ~2 years; we query by date to avoid overlap.
# Only 2018+ needed (study starts 2020-01, price features use 12-month windows).
NADAC_YEAR_IDS = {
    2018: "8de1b213-73c5-552b-b84e-ac795f34d056",
    2019: "76a1984a-6d69-5e4d-86c8-65eb31f0506d",
    2020: "c933dc16-7de9-52b6-8971-4b75992673e0",
    2021: "d5eaf378-dcef-5779-83de-acdd8347d68e",
    2022: "dfa2ab14-06c2-457a-9e36-5cb6d80f8d93",
    2023: "4a00010a-132b-4e4d-a611-543c9521280f",
    2024: "99315a95-37ac-4eee-946a-3c523b4c481e",
    2025: "f38d0706-1239-442c-a3cc-40ef1b686ac0",
    2026: "fbb83258-11c7-47f5-8b18-5f8e79f7e704",
}
DKAN_BASE = "https://data.medicaid.gov/api/1/datastore/query/{dataset_id}/0"
RAW_PATH = DATA_RAW / "nadac_raw.parquet"
OUT_PATH = DATA_PROCESSED / "nadac.parquet"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=60))
def _fetch_date(dataset_id: str, date_str: str) -> pd.DataFrame:
    """Download all NADAC rows for one specific effective_date (~400 rows, fits in one page)."""
    r = requests.get(
        DKAN_BASE.format(dataset_id=dataset_id),
        params={
            "limit": 5000,
            "format": "csv",
            "conditions[0][property]": "effective_date",
            "conditions[0][value]": date_str,
            "conditions[0][operator]": "=",
        },
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def _find_first_date_in_month(dataset_id: str, year_month: str) -> str | None:
    """Return the first available effective_date in YYYY-MM format, or None if no data."""
    r = requests.get(
        DKAN_BASE.format(dataset_id=dataset_id),
        params={
            "limit": 1,
            "conditions[0][property]": "effective_date",
            "conditions[0][value]": f"{year_month}%",
            "conditions[0][operator]": "LIKE",
            "sort": "effective_date",
            "sortOrder": "asc",
        },
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if results:
        return results[0].get("effective_date", "")[:10]
    return None


def ingest_nadac(force: bool = False) -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not force:
        df_raw = pd.read_parquet(RAW_PATH)
        if len(df_raw) > 0:
            print(f"Using cached NADAC data: {RAW_PATH} ({len(df_raw):,} rows)")
        else:
            print("Cached NADAC parquet empty — re-downloading...")
            force = True

    if not RAW_PATH.exists() or force:
        print("Downloading NADAC (one snapshot per month, 2018–present)...")
        frames = []
        for year, dataset_id in sorted(NADAC_YEAR_IDS.items()):
            year_rows = 0
            for month in range(1, 13):
                ym = f"{year}-{month:02d}"
                try:
                    date_str = _find_first_date_in_month(dataset_id, ym)
                    if not date_str:
                        continue
                    df_m = _fetch_date(dataset_id, date_str)
                    if len(df_m) == 0:
                        continue
                    df_m["_year_month"] = ym
                    frames.append(df_m)
                    year_rows += len(df_m)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  WARN {ym}: {e}")
                    time.sleep(1)
            print(f"  {year}: {year_rows:,} rows")
            time.sleep(0.5)

        df_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # Cast any mixed-type object columns to string before parquet serialization
        for col in df_raw.select_dtypes(include="object").columns:
            df_raw[col] = df_raw[col].astype(str)
        df_raw.to_parquet(RAW_PATH, index=False)
        print(f"Raw NADAC saved: {len(df_raw):,} rows")

    # Normalize column names
    df = df_raw.copy()
    rename_map = {
        "NDC Description": "drug_name",
        "NDC": "ndc",
        "NADAC Per Unit": "nadac_per_unit",
        "Effective Date": "effective_date",
        "Pricing Unit": "pricing_unit",
        "OTC": "otc_indicator",
        "Pharmacy Type Indicator": "pharmacy_type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Drop supplemental columns not needed for features
    drop_cols = [
        "Explanation Code", "Classification for Rate Setting",
        "Corresponding Generic Drug NADAC Per Unit", "Corresponding Generic Drug Effective Date",
        "As of Date",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    needed = {"ndc", "nadac_per_unit", "effective_date"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  WARN: missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")

    if "nadac_per_unit" in df.columns:
        df["nadac_per_unit"] = pd.to_numeric(df["nadac_per_unit"], errors="coerce")
    if "effective_date" in df.columns:
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    if "ndc" in df.columns:
        df["ndc"] = df["ndc"].astype(str).str.zfill(11)

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} NADAC rows → {OUT_PATH}")
    if "effective_date" in df.columns and df["effective_date"].notna().any():
        print(f"  Date range: {df['effective_date'].min().date()} → {df['effective_date'].max().date()}")
    return df
