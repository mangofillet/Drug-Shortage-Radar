"""Phase 1b — Drugs@FDA applications (bulk download)."""

import json
import zipfile
import io
import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_RAW, DATA_PROCESSED

DRUGSFDA_INDEX = "https://api.fda.gov/download.json"
RAW_PATH = DATA_RAW / "drugsfda_bulk.json"
OUT_PATH = DATA_PROCESSED / "drugsfda.parquet"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _fetch(url: str) -> requests.Response:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r


def ingest_drugsfda(force: bool = False) -> pd.DataFrame:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not force:
        print(f"Using cached Drugs@FDA data: {RAW_PATH}")
        with open(RAW_PATH) as f:
            all_results = json.load(f)
    else:
        r = _fetch(DRUGSFDA_INDEX)
        index = r.json()
        partitions = index["results"]["drug"]["drugsfda"]["partitions"]
        urls = [p["file"] for p in partitions]
        print(f"Downloading {len(urls)} Drugs@FDA partition(s)...")
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
        app_no = rec.get("application_number", "")
        sponsor = rec.get("sponsor_name", "")
        products = rec.get("products", [{}])
        submissions = rec.get("submissions", [])
        # Earliest approval date
        approval_dates = [
            s.get("submission_status_date", "")
            for s in submissions
            if s.get("submission_type") == "ORIG"
            and s.get("submission_status") in ("AP", "TA")
        ]
        first_approval = min(approval_dates) if approval_dates else ""
        for prod in products:
            rows.append({
                "application_number": app_no,
                "sponsor_name": sponsor,
                "brand_name": prod.get("brand_name", ""),
                "generic_name": prod.get("active_ingredients", [{}])[0].get("name", "") if prod.get("active_ingredients") else "",
                "dosage_form": prod.get("dosage_form", ""),
                "route": prod.get("route", ""),
                "marketing_status": prod.get("marketing_status", ""),
                "te_code": prod.get("te_code", ""),
                "first_approval_date": first_approval,
            })

    df = pd.DataFrame(rows)
    df["first_approval_date"] = pd.to_datetime(df["first_approval_date"], format="%Y%m%d", errors="coerce")

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} Drugs@FDA product rows → {OUT_PATH}")
    app_types = df["application_number"].str[:3].value_counts().head(5)
    print(f"  Application types: {app_types.to_dict()}")
    return df
