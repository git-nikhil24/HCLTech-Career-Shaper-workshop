"""
Electric Vehicle Charging Station Analysis Using Python
-------------------------------------------------------
Student    : Nikhil Pitamber (Roll No. 2026B01010746)
Instructor : Mr. Avinay
Institution: ABES Engineering College, Ghaziabad
Programme  : HCL Career Shaper Program

Input  : 26_ev_charging_sessions.csv  (synthetic dataset - findings describe this sample only)
Libs   : pandas, numpy  (matplotlib optional, for charts)

Run    : python ev_charging_analysis.py [path_to_csv]

Maps to SRS requirements FR-1.1 ... FR-3.7 and outputs OUT-1 ... OUT-5.
"""

import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIGURATION (documented rules - SRS 3.4)
# --------------------------------------------------------------------------
CSV_PATH = "26_ev_charging_sessions.csv"

# FR-3.1: fixed charger types stored in a tuple
CHARGER_TYPES = ("AC Level 2", "DC Fast", "DC Ultra-Fast")

# FR-3.5: duration group cut-offs (minutes), applied consistently
SHORT_MAX = 60      # Short  : duration <  60 min
MEDIUM_MAX = 180    # Medium : 60 <= duration <= 180 min ; Long: > 180 min

EXPECTED_COLUMNS = [
    "Session_ID", "Station_ID", "Station_Location", "Charger_Type",
    "Energy_kWh", "Charging_Duration_Mins", "Charging_Start_Time",
]

SHOW_CHARTS = False   # set True to display Matplotlib charts (optional)


def heading(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# MODULE 1: DATASET LOADING AND DATA QUALITY
# --------------------------------------------------------------------------
def load_data(path):
    """FR-1.1: load CSV with a clear error if missing/unreadable."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: File '{path}' not found. Place it in the project folder "
                 f"or pass the path as an argument.")
    except Exception as exc:
        sys.exit(f"ERROR: Could not read '{path}': {exc}")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: Missing expected columns: {missing_cols}")
    return df


def inspect_data(df):
    """FR-1.2: structure inspection."""
    heading("1. DATASET INSPECTION")
    print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    print("\nColumn names:", list(df.columns))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nDescriptive statistics (numeric columns):")
    print(df.describe().round(2).to_string())


def data_quality_report(df):
    """FR-1.3 & FR-1.4: missing values and duplicate Session_IDs (before cleaning)."""
    heading("2. DATA QUALITY SUMMARY (before cleaning)  [OUT-1]")
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing.to_string())
    print(f"\nMissing Energy_kWh            : {missing['Energy_kWh']}")
    print(f"Missing Charging_Duration_Mins: {missing['Charging_Duration_Mins']}")

    dup_count = df["Session_ID"].duplicated().sum()
    print(f"\nDuplicate Session_ID records (extra copies): {dup_count}")
    if dup_count:
        print("Duplicated Session_IDs:", sorted(df.loc[df['Session_ID'].duplicated(), 'Session_ID'].unique()))
    return dup_count


def clean_data(df):
    """
    FR-1.5 / SRS 3.4 cleaning rules (original DataFrame is NOT modified):
      1. Duplicates : keep the first record per Session_ID.
      2. Missing Energy_kWh / Charging_Duration_Mins : median imputation WITHIN each
         Charger_Type group (energy and duration depend strongly on charger type, so
         a group median is more realistic than a single global median).
      3. Invalid timestamps are reported and excluded from time analysis only.
    """
    heading("3. DATA CLEANING ACTIONS")
    cleaned = df.copy()
    log = []
