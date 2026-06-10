from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_WAYBACK = DATA_RAW / "wayback"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Set after Phase 1a label depth verification; fallback to 2020-01 if HTML parsing fails
STUDY_START = "2020-01"

# Temporal splits — adjust START only if reconstruction window differs; keep structure
TRAIN_END = "2022-12"
VAL_START = "2023-01"
VAL_END = "2024-12"
TEST_START = "2025-01"
TEST_END = "2025-11"  # last month with a complete 6-month label horizon as of June 2026

LABEL_HORIZON_MONTHS = 6
LABEL_HORIZON_SHORT_MONTHS = 3  # sensitivity analysis only

# Drug universe filter
MARKETING_CATEGORIES = {"NDA", "ANDA", "BLA"}

# Entity resolution
FUZZY_MATCH_THRESHOLD = 92  # rapidfuzz token_set_ratio cutoff
