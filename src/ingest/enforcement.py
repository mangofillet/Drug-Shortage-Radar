"""Phase 1e — openFDA drug enforcement/recall records (bulk download)."""

import json
import zipfile
import io
import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_RAW, DATA_PROCESSED

ENFORCEMENT_INDEX = "https://api.fda.gov/download.json"
RAW_PATH = DATA_RAW / "enforcement_bulk.json"
OUT_PATH = DATA_PROCESSED / "enforcement.parquet"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _fetch(url: str) -> requests.Response:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r


def ingest_enforcement(force: bool = False) -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not force:
        print(f"Using cached enforcement data: {RAW_PATH}")
        with open(RAW_PATH) as f:
            all_results = json.load(f)
    else:
        r = _fetch(ENFORCEMENT_INDEX)
        index = r.json()
        partitions = index["results"]["drug"]["enforcement"]["partitions"]
        urls = [p["file"] for p in partitions]
        print(f"Downloading {len(urls)} enforcement partition(s)...")
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
            "recall_number": rec.get("recall_number", ""),
            "recalling_firm": rec.get("recalling_firm", ""),
            "recall_initiation_date": rec.get("recall_initiation_date", ""),
            "recall_class": rec.get("classification", ""),
            "product_description": rec.get("product_description", ""),
            "reason_for_recall": rec.get("reason_for_recall", ""),
            "status": rec.get("status", ""),
            "product_type": rec.get("product_type", ""),
            "voluntary_mandated": rec.get("voluntary_mandated", ""),
            "distribution_pattern": rec.get("distribution_pattern", ""),
            "city": rec.get("city", ""),
            "state": rec.get("state", ""),
            "country": rec.get("country", ""),
        })

    df = pd.DataFrame(rows)
    df["recall_initiation_date"] = pd.to_datetime(df["recall_initiation_date"], format="%Y%m%d", errors="coerce")

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} enforcement rows → {OUT_PATH}")
    print(f"  Date range: {df['recall_initiation_date'].min().date()} → {df['recall_initiation_date'].max().date()}")
    print(f"  Class breakdown: {df['recall_class'].value_counts().to_dict()}")
    return df
