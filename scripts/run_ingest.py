"""
Phase 1: Run all ingesters in order.
Idempotent — skips cached files unless --force.
Usage: python scripts/run_ingest.py [--force]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.shortages import reconstruct_shortage_history, fetch_live_openfda
from src.ingest.ndc import ingest_ndc
from src.ingest.drugsfda import ingest_drugsfda
from src.ingest.enforcement import ingest_enforcement
from src.ingest.nadac import ingest_nadac
from src.ingest.inspections import ingest_inspections
from src.ingest.compliance import ingest_compliance


def main(force: bool = False):
    print("=" * 60)
    print("PHASE 1: DATA INGESTION")
    print("=" * 60)

    print("\n--- 1a: Shortage labels (Wayback reconstruction) ---")
    shortages = reconstruct_shortage_history(force=force)
    fetch_live_openfda(force=force)

    print("\n--- 1b: NDC directory ---")
    ndc = ingest_ndc(force=force)

    print("\n--- 1b: Drugs@FDA applications ---")
    drugsfda = ingest_drugsfda(force=force)

    print("\n--- 1c: Inspections + citations ---")
    insp, cites = ingest_inspections(force=force)

    print("\n--- 1d: Compliance actions ---")
    compliance = ingest_compliance(force=force)

    print("\n--- 1e: Enforcement / recalls ---")
    enforcement = ingest_enforcement(force=force)

    print("\n--- 1f: NADAC prices ---")
    nadac = ingest_nadac(force=force)

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"Shortage events reconstructed : {len(shortages):>8,}")
    print(f"NDC product rows              : {len(ndc):>8,}")
    print(f"Drugs@FDA product rows        : {len(drugsfda):>8,}")
    print(f"Inspection rows               : {len(insp):>8,}")
    print(f"Citation rows                 : {len(cites):>8,}")
    print(f"Compliance action rows        : {len(compliance):>8,}")
    print(f"Enforcement/recall rows       : {len(enforcement):>8,}")
    print(f"NADAC price rows              : {len(nadac):>8,}")
    print("\nPhase 1 complete. Verify counts above before proceeding to Phase 2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    main(force=args.force)
