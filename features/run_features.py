#To accomplish your task of creating a Python script for feature engineering, you'll want to follow these steps:

# 1. Load the necessary libraries.
# 2. Load your cleaned data (`data/cleaned.csv`).
# 3. Define or import your `create_features` and `select_features` functions.
# 4. Perform feature engineering using these functions.
# 5. Save the transformed data to a new CSV file (`data/features.csv`).
# 6. Print the before and after shape of the data and list any features that were kept.
# 7. Measure and print the time taken for the entire operation.


import os
import sys
import time
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# Import the real functions from engineering.py.
# sys.path insert ensures Python finds engineering.py when this script is
# run from the project root (my-ml-project/).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_engineering import create_features, select_features
 
 
# ---------------------------------------------------------------------------
# CONFIGURATION — edit paths here if your layout differs
# ---------------------------------------------------------------------------
INPUT_PATH  = "data/cleaned.csv"
OUTPUT_PATH = "data/features.csv"
TARGET_COL  = "Class"
 
 
# ---------------------------------------------------------------------------
# HELPER — pretty section banner
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
 
 
# ---------------------------------------------------------------------------
# HELPER — print kept features in a clean numbered list
# ---------------------------------------------------------------------------
def print_kept_features(features: list[str]) -> None:
    print(f"\n  {'#':<5} {'Feature'}")
    print(f"  {'-'*5} {'-'*35}")
    for i, feat in enumerate(features, 1):
        print(f"  {i:<5} {feat}")
 
 
# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
 
    pipeline_start = time.time()
 
    # -----------------------------------------------------------------------
    # STEP 1 — Load data
    # -----------------------------------------------------------------------
    banner("STEP 1 — Load cleaned data")
    t0 = time.time()
 
    if not os.path.exists(INPUT_PATH):
        print(f"\n  ❌ ERROR: File not found → {INPUT_PATH}")
        print(  "     Make sure data/cleaned.csv exists in your project root.")
        print(  "     If your file is named differently, update INPUT_PATH in")
        print(  "     this script (line ~37).")
        sys.exit(1)
 
    df_raw = pd.read_csv(INPUT_PATH)
    elapsed = time.time() - t0
 
    print(f"\n  Source file   : {INPUT_PATH}")
    print(f"  Rows loaded   : {df_raw.shape[0]:,}")
    print(f"  Columns       : {df_raw.shape[1]}")
    print(f"  Shape         : {df_raw.shape}")
    print(f"  ⏱  Loaded in {elapsed:.2f}s")
 
    # -----------------------------------------------------------------------
    # STEP 2 — Feature engineering
    # -----------------------------------------------------------------------
    banner("STEP 2 — Feature engineering  (create_features)")
    t0 = time.time()
 
    df_engineered = create_features(df_raw)
    elapsed = time.time() - t0
 
    new_cols = [c for c in df_engineered.columns if c not in df_raw.columns]
 
    print(f"\n  Shape before  : {df_raw.shape}")
    print(f"  Shape after   : {df_engineered.shape}")
    print(f"  Features added: {len(new_cols)}")
    print(f"\n  New columns created:")
    for col in new_cols:
        nulls = df_engineered[col].isna().sum()
        print(f"    ✓ {col:<35} nulls={nulls}")
    print(f"\n  ⏱  Engineered in {elapsed:.2f}s")
 
    # -----------------------------------------------------------------------
    # STEP 3 — Feature selection
    # -----------------------------------------------------------------------
    banner("STEP 3 — Feature selection  (select_features)")
    t0 = time.time()
 
    selected_features, df_selected = select_features(
        df_engineered,
        target_col=TARGET_COL,
        corr_threshold=0.95,
        variance_threshold_pct=0.01,
    )
    elapsed = time.time() - t0
 
    features_before = [c for c in df_engineered.columns if c != TARGET_COL]
    features_dropped = [f for f in features_before if f not in selected_features]
 
    print(f"\n  Features before selection : {len(features_before)}")
    print(f"  Features dropped          : {len(features_dropped)}")
    print(f"  Features kept             : {len(selected_features)}")
    print(f"  Final shape (with target) : {df_selected.shape}")
    print(f"\n  ⏱  Selected in {elapsed:.2f}s")
 
    # -----------------------------------------------------------------------
    # STEP 4 — Print full list of kept features
    # -----------------------------------------------------------------------
    banner("KEPT FEATURES (inputs to model)")
    print_kept_features(selected_features)
    print(f"\n  Target column kept        : {TARGET_COL}")
 
    # -----------------------------------------------------------------------
    # STEP 5 — Save output
    # -----------------------------------------------------------------------
    banner("STEP 4 — Save features to disk")
    t0 = time.time()
 
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_selected.to_csv(OUTPUT_PATH, index=False)
    elapsed = time.time() - t0
 
    print(f"\n  Saved to      : {OUTPUT_PATH}")
    print(f"  Rows saved    : {df_selected.shape[0]:,}")
    print(f"  Columns saved : {df_selected.shape[1]}")
    print(f"  ⏱  Saved in {elapsed:.2f}s")
 
    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    banner("PIPELINE COMPLETE")
    print(f"\n  {'Input file':<30} {INPUT_PATH}")
    print(f"  {'Output file':<30} {OUTPUT_PATH}")
    print(f"  {'Rows':<30} {df_selected.shape[0]:,}")
    print(f"  {'Raw columns (before)':<30} {df_raw.shape[1]}")
    print(f"  {'After engineering':<30} {df_engineered.shape[1]}")
    print(f"  {'After selection (excl. target)':<30} {len(selected_features)}")
    print(f"  {'Total elapsed':<30} {total_elapsed:.2f}s")
    print()
 
 
# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()





