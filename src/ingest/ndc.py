"""Phase 1b — openFDA NDC directory (bulk download)."""

import json
import zipfile
import io
import time
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_RAW, DATA_PROCESSED

NDC_DOWNLOAD_INDEX = "https://api.fda.gov/download.json"
RAW_PATH = DATA_RAW / "ndc_bulk.json"
OUT_PATH = DATA_PROCESSED / "ndc.parquet"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _fetch(url: str) -> requests.Response:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r


def _get_bulk_urls() -> list[str]:
    r = _fetch(NDC_DOWNLOAD_INDEX)
    index = r.json()
    partitions = index["results"]["drug"]["ndc"]["partitions"]
    return [p["file"] for p in partitions]


def ingest_ndc(force: bool = False) -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not force:
        print(f"Using cached NDC data: {RAW_PATH}")
        with open(RAW_PATH) as f:
            all_results = json.load(f)
    else:
        urls = _get_bulk_urls()
        print(f"Downloading {len(urls)} NDC bulk partition(s)...")
        all_results = []
        for url in urls:
            r = _fetch(url)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                fname = [n for n in z.namelist() if n.endswith(".json")][0]
                data = json.loads(z.read(fname))
                all_results.extend(data.get("results", []))
            time.sleep(0.5)
        with open(RAW_PATH, "w") as f:
            json.dump(all_results, f)

    rows = []
    for rec in all_results:
        rows.append({
            "product_ndc": rec.get("product_ndc", ""),
            "generic_name": rec.get("generic_name", ""),
            "brand_name": rec.get("brand_name", ""),
            "labeler_name": rec.get("labeler_name", ""),
            "dosage_form": rec.get("dosage_form", ""),
            "route": str(rec.get("route", [])),
            "marketing_category": rec.get("marketing_category", ""),
            "application_number": rec.get("application_number", ""),
            "dea_schedule": rec.get("dea_schedule", ""),
            "marketing_start_date": rec.get("marketing_start_date", ""),
            "marketing_end_date": rec.get("marketing_end_date", ""),
            "product_type": rec.get("product_type", ""),
            "listing_expiration_date": rec.get("listing_expiration_date", ""),
        })

    df = pd.DataFrame(rows)

    # Parse dates
    for col in ("marketing_start_date", "marketing_end_date", "listing_expiration_date"):
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} NDC rows → {OUT_PATH}")
    print(f"  Date range: {df['marketing_start_date'].min().date()} → {df['marketing_start_date'].max().date()}")
    print(f"  Marketing categories: {df['marketing_category'].value_counts().head(5).to_dict()}")
    return df
