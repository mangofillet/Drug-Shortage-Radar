"""
Phase 1a — Drug shortage label reconstruction via Wayback Machine snapshots.

Key verified finding: openFDA shortages endpoint is a live snapshot only (resolved records
purged). Historical labels reconstructed from Wayback Machine captures of FDA shortage pages.

Sources:
  CSV: accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm (96 captures, 2019-10+)
  HTML: accessdata.fda.gov/scripts/drugshortages/default.cfm (~2400 captures, 2014+, dense 2020+)
"""

import csv
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_WAYBACK, DATA_PROCESSED

CDX_API = "http://web.archive.org/cdx/search/cdx"
WB_FETCH = "https://web.archive.org/web/{ts}id_/{url}"

CSV_URL = "https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm"
HTML_URL = "https://www.accessdata.fda.gov/scripts/drugshortages/default.cfm"
OPENFDA_URL = "https://api.fda.gov/drug/shortages.json"

DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def _parse_fda_date(s: str) -> pd.Timestamp | None:
    if not s or not s.strip():
        return None
    m = DATE_RE.search(s.strip())
    if m:
        mo, dy, yr = m.groups()
        try:
            return pd.Timestamp(f"{yr}-{mo.zfill(2)}-{dy.zfill(2)}")
        except Exception:
            return None
    return None


def _normalize_key(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60))
def _fetch(url: str, **kwargs) -> requests.Response:
    r = requests.get(url, timeout=60, **kwargs)
    if r.status_code in (429, 503, 504):
        r.raise_for_status()  # triggers retry
    r.raise_for_status()
    return r


def _cdx_snapshots(target_url: str, collapse_months: int = 2) -> list[dict]:
    """Return list of {ts, url} collapsed to ~N snapshots/month."""
    # Cache CDX results to avoid re-hitting the API
    cache_key = target_url.replace("/", "_").replace(":", "").replace(".", "_")[:80]
    cache_path = DATA_WAYBACK / f"cdx_{cache_key}.json"
    DATA_WAYBACK.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            rows = json.load(f)
    else:
        params = {
            "url": target_url,
            "fl": "timestamp,original",
            "filter": "statuscode:200",
            "output": "json",
        }
        r = _fetch(CDX_API, params=params)
        rows = r.json()
        with open(cache_path, "w") as f:
            json.dump(rows, f)
    if len(rows) <= 1:
        return []
    header, *data = rows
    ts_idx = header.index("timestamp")
    url_idx = header.index("original")
    # Collapse to ≤2 per month by keeping first and middle of each month's captures
    by_month: dict[str, list] = {}
    for row in data:
        month = row[ts_idx][:6]
        by_month.setdefault(month, []).append(row)
    result = []
    for month, captures in sorted(by_month.items()):
        picks = [captures[0]]
        if len(captures) > 2:
            picks.append(captures[len(captures) // 2])
        for c in picks[:collapse_months]:
            result.append({"ts": c[ts_idx], "url": c[url_idx]})
    return result


def _wb_path(ts: str, kind: str) -> Path:
    return DATA_WAYBACK / f"{kind}_{ts}.raw"


def _download_snapshot(ts: str, original_url: str, kind: str, force: bool = False) -> Path | None:
    path = _wb_path(ts, kind)
    if path.exists() and not force:
        return path
    fetch_url = WB_FETCH.format(ts=ts, url=original_url)
    try:
        r = _fetch(fetch_url, headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(r.content)
        time.sleep(1.0)  # polite: ≤1 req/sec to archive.org
        return path
    except Exception as e:
        print(f"  WARN: failed {ts}: {e}")
        return None


def _parse_csv_snapshot(raw: bytes) -> list[dict]:
    """Parse FDA shortage CSV snapshot → list of record dicts."""
    # Strip leading blank lines — some CSV snapshots start with \r\n before the header
    text = raw.decode("utf-8-sig", errors="replace").lstrip()
    reader = csv.DictReader(io.StringIO(text))
    # Strip whitespace from headers
    reader.fieldnames = [f.strip() for f in (reader.fieldnames or [])]
    records = []
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items() if k}
        records.append({
            "generic_name": row.get("Generic Name", ""),
            "company_name": row.get("Company Name", ""),
            "presentation": row.get("Presentation", ""),
            "status": row.get("Status", ""),
            "initial_posting_date": row.get("Initial Posting Date", ""),
            "update_date": row.get("Date of Update", ""),
            "discontinued_date": row.get("Date Discontinued", ""),
            "therapeutic_category": row.get("Therapeutic Category", ""),
            "availability": row.get("Availability Information", ""),
            "shortage_reason": row.get("Reason for Shortage", ""),
            "source": "csv",
        })
    return records


def _parse_html_snapshot(raw: bytes) -> list[dict]:
    """Parse FDA shortage HTML page → list of record dicts."""
    soup = BeautifulSoup(raw, "lxml")
    records = []
    # The page uses tabs; each drug appears as a row with a link to its detail page.
    # The initial_posting_date is in the detail page, but the drug name + status are here.
    # Extract from the tabs-1 (Current/Resolved) and tabs-2 (Discontinuations) panels.
    for tab_id in ("tabs-1", "tabs-2"):
        panel = soup.find("div", {"id": tab_id})
        if not panel:
            continue
        status_tag = "Current" if tab_id == "tabs-1" else "To Be Discontinued"
        for a in panel.find_all("a", href=True):
            href = a.get("href", "")
            if "dsp_ActiveIngredientDetails" not in href:
                continue
            # Extract AI param (generic name + dosage form)
            ai_match = re.search(r"AI=([^&]+)", href)
            st_match = re.search(r"st=([^&]+)", href)
            if not ai_match:
                continue
            ai = unquote_plus(ai_match.group(1))
            st = st_match.group(1) if st_match else ""
            # Derive status from tab
            if st == "c":
                status = "Current"
            elif st == "d":
                status = "To Be Discontinued"
            else:
                status = status_tag
            records.append({
                "generic_name": ai,
                "company_name": "",
                "presentation": "",
                "status": status,
                "initial_posting_date": "",
                "update_date": "",
                "discontinued_date": "",
                "therapeutic_category": "",
                "availability": "",
                "shortage_reason": "",
                "source": "html",
            })
    # Deduplicate by name+status within this snapshot
    seen = set()
    unique = []
    for r in records:
        k = (r["generic_name"], r["status"])
        if k not in seen:
            seen.add(k)
            unique.append(r)
    return unique


def reconstruct_shortage_history(force: bool = False) -> pd.DataFrame:
    """
    Main function. Downloads Wayback snapshots and reconstructs shortage onset history.
    Returns DataFrame with one row per (generic_name, company_name, presentation, onset_date).
    """
    DATA_WAYBACK.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # --- Step 1: gather CDX snapshot lists ---
    print("Fetching CDX snapshot list for CSV URL...")
    csv_snaps = _cdx_snapshots(CSV_URL)
    print(f"  CSV snapshots to download: {len(csv_snaps)}")

    print("Fetching CDX snapshot list for HTML URL...")
    html_snaps = _cdx_snapshots(HTML_URL)
    print(f"  HTML snapshots to download: {len(html_snaps)}")

    # --- Step 2: download and parse CSV snapshots (primary, cleaner data) ---
    all_records = []
    print(f"\nDownloading {len(csv_snaps)} CSV snapshots...")
    for i, snap in enumerate(csv_snaps):
        path = _download_snapshot(snap["ts"], snap["url"], "csv", force=force)
        if path and path.exists():
            recs = _parse_csv_snapshot(path.read_bytes())
            for r in recs:
                r["snapshot_ts"] = snap["ts"]
            all_records.extend(recs)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(csv_snaps)} CSV snapshots processed")

    # --- Step 3: download and parse HTML snapshots (gap-fill + pre-2019) ---
    # Only download HTML for months not already covered by CSV (and pre-2019)
    csv_months = {s["ts"][:6] for s in csv_snaps}
    html_needed = [s for s in html_snaps if s["ts"][:6] not in csv_months or s["ts"][:4] < "2019"]
    print(f"\nDownloading {len(html_needed)} HTML snapshots (gap-fill + pre-2019)...")
    for i, snap in enumerate(html_needed):
        path = _download_snapshot(snap["ts"], snap["url"], "html", force=force)
        if path and path.exists():
            recs = _parse_html_snapshot(path.read_bytes())
            for r in recs:
                r["snapshot_ts"] = snap["ts"]
            all_records.extend(recs)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(html_needed)} HTML snapshots processed")

    if not all_records:
        raise RuntimeError("No records parsed from any snapshot — check network/CDX access")

    df = pd.DataFrame(all_records)

    # --- Step 4: parse dates ---
    df["initial_posting_date_parsed"] = df["initial_posting_date"].apply(_parse_fda_date)
    df["update_date_parsed"] = df["update_date"].apply(_parse_fda_date)
    df["discontinued_date_parsed"] = df["discontinued_date"].apply(_parse_fda_date)
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_ts"], format="%Y%m%d%H%M%S", errors="coerce")

    # --- Step 5: reconstruct onset events ---
    # For CSV records with initial_posting_date, onset is exact.
    # For HTML records (no date), onset is approximated as the snapshot month.
    df["generic_name_norm"] = df["generic_name"].apply(_normalize_key)

    # Priority: CSV records (have exact dates) trump HTML records.
    csv_df = df[df["source"] == "csv"].copy()
    html_df = df[df["source"] == "html"].copy()

    # Build onset table from CSV: deduplicate by (generic_name_norm, presentation_norm, company_norm)
    csv_df["presentation_norm"] = csv_df["presentation"].apply(_normalize_key)
    csv_df["company_norm"] = csv_df["company_name"].apply(_normalize_key)

    onsets_csv = (
        csv_df[csv_df["initial_posting_date_parsed"].notna()]
        .groupby(["generic_name_norm", "presentation_norm", "company_norm"], as_index=False)
        .agg(
            generic_name=("generic_name", "first"),
            company_name=("company_name", "first"),
            presentation=("presentation", "first"),
            therapeutic_category=("therapeutic_category", "first"),
            shortage_reason=("shortage_reason", "first"),
            initial_posting_date=("initial_posting_date_parsed", "min"),
            last_seen_date=("snapshot_dt", "max"),
            status_last=("status", "last"),
        )
        .rename(columns={"initial_posting_date": "onset_date"})
    )
    onsets_csv["source"] = "csv_wayback"

    # Build onset table from HTML: drug appeared in snapshot = was in shortage that month
    # Use snapshot_dt as approximate onset floor (may miss exact date)
    html_df["company_norm"] = ""
    html_df["presentation_norm"] = ""
    onsets_html = (
        html_df.groupby(["generic_name_norm"], as_index=False)
        .agg(
            generic_name=("generic_name", "first"),
            company_name=("company_name", "first"),
            presentation=("presentation", "first"),
            therapeutic_category=("therapeutic_category", "first"),
            shortage_reason=("shortage_reason", "first"),
            onset_date=("snapshot_dt", "min"),
            last_seen_date=("snapshot_dt", "max"),
            status_last=("status", "last"),
        )
    )
    onsets_html["source"] = "html_wayback"
    onsets_html["company_norm"] = ""
    onsets_html["presentation_norm"] = ""

    # Merge: drop HTML records where CSV already covers the same drug
    csv_names = set(onsets_csv["generic_name_norm"].unique())
    onsets_html_new = onsets_html[~onsets_html["generic_name_norm"].isin(csv_names)]

    onsets = pd.concat([onsets_csv, onsets_html_new], ignore_index=True)
    onsets = onsets.sort_values("onset_date").reset_index(drop=True)

    # Approximate resolution date = last_seen_date + 2 months (interval-censored upper bound)
    onsets["resolved_date_approx"] = onsets.apply(
        lambda r: r["last_seen_date"] + pd.DateOffset(months=2)
        if r["status_last"] in ("Resolved",) else pd.NaT,
        axis=1,
    )

    out_path = DATA_PROCESSED / "shortages_raw.parquet"
    onsets.to_parquet(out_path, index=False)
    print(f"\nSaved {len(onsets)} shortage events → {out_path}")

    # Print year breakdown
    year_counts = (
        onsets[onsets["onset_date"].notna()]
        .assign(year=onsets["onset_date"].dt.year)
        .groupby("year")
        .size()
    )
    print("\nOnset events by year:")
    for yr, cnt in year_counts.items():
        print(f"  {yr}: {cnt}")

    return onsets


def fetch_live_openfda(force: bool = False) -> pd.DataFrame:
    """Pull live openFDA shortages endpoint for openfda enrichment block."""
    out_path = DATA_PROCESSED / "shortages_live.parquet"
    raw_path = DATA_WAYBACK.parent / "shortages_live.json"
    if raw_path.exists() and not force:
        print(f"Using cached live feed: {raw_path}")
        with open(raw_path) as f:
            all_results = json.load(f)
    else:
        all_results = []
        skip = 0
        limit = 1000
        while True:
            url = f"{OPENFDA_URL}?limit={limit}&skip={skip}"
            r = _fetch(url)
            data = r.json()
            results = data.get("results", [])
            all_results.extend(results)
            total = data["meta"]["results"]["total"]
            skip += limit
            if skip >= total or skip >= 25000:
                break
            time.sleep(0.25)
        with open(raw_path, "w") as f:
            json.dump(all_results, f)

    rows = []
    for rec in all_results:
        openfda = rec.get("openfda", {})
        rows.append({
            "generic_name": rec.get("generic_name", ""),
            "package_ndc": rec.get("package_ndc", ""),
            "status": rec.get("status", ""),
            "initial_posting_date": rec.get("initial_posting_date", ""),
            "availability": rec.get("availability", ""),
            "company_name": rec.get("company_name", ""),
            "therapeutic_category": str(rec.get("therapeutic_category", "")),
            "dosage_form": rec.get("dosage_form", ""),
            "application_number": str(openfda.get("application_number", [])),
            "manufacturer_name": str(openfda.get("manufacturer_name", [])),
            "product_ndc": str(openfda.get("product_ndc", [])),
            "route": str(openfda.get("route", [])),
            "substance_name": str(openfda.get("substance_name", [])),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} live openFDA rows → {out_path}")
    return df
