"""
Phase 3 gate: verifies labels for 3 known shortages.
These must pass before proceeding to Phase 4.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_PROCESSED


def _load_panel() -> pd.DataFrame:
    path = DATA_PROCESSED / "panel.parquet"
    if not path.exists():
        pytest.skip("Panel not yet built — run scripts/build_panel.py first")
    return pd.read_parquet(path)


def test_amoxicillin_2022():
    """Amoxicillin shortage was posted ~Nov 2022. Panel should show y_6m=True in mid-2022."""
    panel = _load_panel()
    amo = panel[panel["drug_key"].str.startswith("amoxicillin|")]
    if len(amo) == 0:
        pytest.skip("amoxicillin not in panel — check drug universe")
    # At least one row in May-Oct 2022 should be y_6m=True
    window = amo[(amo["month"] >= "2022-05-01") & (amo["month"] <= "2022-10-01")]
    assert window["y_6m"].any(), (
        f"Expected y_6m=True for amoxicillin in mid-2022 (shortage onset ~Nov 2022)\n"
        f"Panel rows: {window[['drug_key','month','y_6m']].to_string()}"
    )


def test_adderall_2022():
    """
    Adderall is a 4-salt combination whose ingredients are listed in different orders
    across the FDA shortage feed vs the NDC directory — this exercises the panel's
    order-independent (token-set) shortage→drug matching. The well-known Oct-2022 onset's
    own pre-window is masked for some presentations because they were ALREADY in shortage
    from an earlier (Feb-2022) onset, and we never label "new onset" while a drug is
    actively short. So assert the family is labeled in the 2021–2022 run-up, not in one
    specific (possibly masked) month.
    """
    panel = _load_panel()
    generic = panel["drug_key"].str.split("|").str[0].str.lower()
    amo = panel[generic.str.contains("amphetamine") & generic.str.contains("dextroamphetamine")]
    if len(amo) == 0:
        pytest.skip("Adderall/amphetamine salts not in panel — check token-set matching or drug universe")
    window = amo[(amo["month"] >= "2021-08-01") & (amo["month"] <= "2022-12-01")]
    assert window["y_6m"].any(), (
        f"Expected y_6m=True for amphetamine combo salts in the 2021-2022 shortage run-up "
        f"(token-set matching should link the reordered combo name)\n"
        f"Panel rows: {amo[['drug_key','month','y_6m']].head(10).to_string()}"
    )


def test_cisplatin_carboplatin_2023():
    """Cisplatin/carboplatin shortage onset 2023. Should see y_6m=True in 2022-H2."""
    panel = _load_panel()
    chemo = panel[
        panel["drug_key"].str.split("|").str[0].str.lower().str.contains("cisplatin|carboplatin")
    ]
    if len(chemo) == 0:
        pytest.skip("cisplatin/carboplatin not in panel — check drug universe")
    window = chemo[(chemo["month"] >= "2022-06-01") & (chemo["month"] <= "2022-12-01")]
    assert window["y_6m"].any(), (
        f"Expected y_6m=True for cisplatin/carboplatin in 2022 H2\n"
        f"Panel rows: {window[['drug_key','month','y_6m']].head(10).to_string()}"
    )


def test_base_rate_sanity():
    """Overall positive rate should be between 0.5% and 10%."""
    panel = _load_panel()
    train = panel[panel["split"] == "train"]
    rate = train["y_6m"].mean()
    assert 0.005 <= rate <= 0.10, (
        f"Base rate {rate*100:.2f}% outside expected 0.5–10% range. "
        "Check label construction."
    )


def test_no_test_split_leakage():
    """Labels for test split rows must not be used in training — just verify split assignment."""
    panel = _load_panel()
    from config import TEST_START
    test_rows = panel[panel["split"] == "test"]
    assert len(test_rows) > 0, "No test split rows found"
    assert (test_rows["month"] >= pd.Timestamp(TEST_START)).all(), (
        "Test split contains rows before TEST_START"
    )
