"""
LOCKED TEST SET EVALUATION — DO NOT RUN UNLESS USER EXPLICITLY REQUESTS IT.

This script evaluates the final frozen model on the 2025-01 → 2025-11 test window.
It will be run exactly ONCE, at the user's explicit request, at the end of Phase 8.
Running it prematurely invalidates the capstone evaluation.
"""
import sys

CONFIRMATION = "I confirm I want to run the locked test evaluation"

if __name__ == "__main__":
    print("=" * 70)
    print("WARNING: This runs the LOCKED test set evaluation.")
    print("This should only be done once, at the user's explicit request.")
    print("=" * 70)
    response = input(f'Type exactly: "{CONFIRMATION}"\n> ')
    if response.strip() != CONFIRMATION:
        print("Aborted.")
        sys.exit(1)
    print("Proceeding with test evaluation...")
    # TODO: implement after Phase 5 model is frozen
