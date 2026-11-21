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

# Rule 1 - duplicates
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset="Session_ID", keep="first").reset_index(drop=True)
    log.append(f"Removed {before - len(cleaned)} duplicate Session_ID rows (kept first).")

    # Rule 2 - group-wise median imputation
    for col in ("Energy_kWh", "Charging_Duration_Mins"):
        n_missing = cleaned[col].isnull().sum()
        if n_missing:
            group_median = cleaned.groupby("Charger_Type")[col].transform("median")
            cleaned[col] = cleaned[col].fillna(group_median)
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())  # safety fallback
        log.append(f"Imputed {n_missing} missing {col} values using Charger_Type median.")

    # Rule 3 - datetime conversion (FR-2.1): invalid values identified, not silently misread
    cleaned["Charging_Start_Time"] = pd.to_datetime(cleaned["Charging_Start_Time"], errors="coerce")
    bad_time = cleaned["Charging_Start_Time"].isnull().sum()
    log.append(f"Invalid/missing timestamps found: {bad_time} "
               f"(excluded from time-based analysis only).")

    for line in log:
        print("-", line)
    print(f"\nRecords after cleaning: {len(cleaned)}")
    return cleaned


# --------------------------------------------------------------------------
# MODULE 2: TIME PROCESSING AND CORE ANALYTICS
# --------------------------------------------------------------------------
def add_time_features(df):
    """FR-2.2: hour, weekday, month from valid datetimes."""
    df = df.copy()
    ts = df["Charging_Start_Time"]
    df["Hour"] = ts.dt.hour
    df["Weekday"] = ts.dt.day_name()
    df["Month"] = ts.dt.month_name()
    return df


def overall_summary(df):
    """FR-2.3 & FR-2.4 [OUT-2]."""
    heading("4. OVERALL SUMMARY  [OUT-2]")
    total_sessions = len(df)
    total_energy = df["Energy_kWh"].sum()
    avg_duration = df["Charging_Duration_Mins"].mean()
    avg_energy = df["Energy_kWh"].mean()
    print(f"Total charging sessions          : {total_sessions}")
    print(f"Total energy delivered           : {total_energy:,.2f} kWh")
    print(f"Average charging duration        : {avg_duration:.2f} minutes")
    print(f"Average energy per session       : {avg_energy:.2f} kWh")
    return total_sessions, total_energy, avg_duration, avg_energy

def station_ranking(df):
    """FR-2.5: rank by session count and total energy."""
    heading("5. STATION RANKING  [OUT-3]")
    stations = (df.groupby(["Station_ID", "Station_Location"])
                  .agg(Sessions=("Session_ID", "count"),
                       Total_Energy_kWh=("Energy_kWh", "sum"),
                       Avg_Energy_kWh=("Energy_kWh", "mean"))
                  .reset_index())
    stations["Rank_by_Sessions"] = stations["Sessions"].rank(ascending=False, method="min").astype(int)
    stations["Rank_by_Energy"] = stations["Total_Energy_kWh"].rank(ascending=False, method="min").astype(int)
    stations = stations.sort_values(["Rank_by_Sessions", "Rank_by_Energy"]).round(2)
    print(stations.to_string(index=False))
    return stations


def charger_comparison(df):
    """FR-2.6: average duration per charger type."""
    heading("6. CHARGER TYPE COMPARISON  [OUT-3]")
    comp = (df.groupby("Charger_Type")
              .agg(Sessions=("Session_ID", "count"),
                   Avg_Duration_Mins=("Charging_Duration_Mins", "mean"),
                   Avg_Energy_kWh=("Energy_kWh", "mean"))
              .round(2).sort_values("Avg_Duration_Mins", ascending=False))
    print(comp.to_string())
    return comp


def location_comparison(df):
    """FR-2.8: energy consumption per Station_Location."""
    heading("7. ENERGY CONSUMPTION BY STATION LOCATION  [OUT-3]")
    loc = (df.groupby("Station_Location")["Energy_kWh"]
             .agg(Total_Energy_kWh="sum", Avg_Energy_kWh="mean")
             .round(2).sort_values("Total_Energy_kWh", ascending=False))
    print(loc.to_string())
    return loc


def demand_patterns(df):
    """FR-2.7: peak hours and busiest weekdays (valid datetimes only)."""
    heading("8. DEMAND PATTERNS  [OUT-4]")
    valid = df.dropna(subset=["Charging_Start_Time"])

    hour_counts = valid["Hour"].value_counts().sort_index()
    top_hours = valid["Hour"].value_counts().head(3)
    print("Top 3 peak charging hours (hour of day : sessions):")
    for h, c in top_hours.items():
        print(f"  {int(h):02d}:00 - {int(h):02d}:59  ->  {c} sessions")
    peak_hour = int(top_hours.index[0])

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = valid["Weekday"].value_counts().reindex(day_order).fillna(0).astype(int)
    print("\nSessions by weekday:")
    print(day_counts.to_string())
    busiest_day = day_counts.idxmax()
    print(f"\nBusiest weekday: {busiest_day} ({day_counts.max()} sessions)")

    print("\nSessions by month:")
    print(valid["Month"].value_counts().to_string())
    return peak_hour, busiest_day, hour_counts, day_counts


# --------------------------------------------------------------------------
# MODULE 3: STATISTICS, OOP AND RECOMMENDATIONS
# --------------------------------------------------------------------------
def station_dictionary(df):
    """FR-3.2: dictionary of station-wise session counts."""
    return df["Station_ID"].value_counts().sort_index().to_dict()


def avg_energy_for_station(df, station_id):
    """FR-3.2: average energy per session for a selected station."""
    subset = df.loc[df["Station_ID"] == station_id, "Energy_kWh"]
    if subset.empty:
        return None
    return float(subset.mean())


def numpy_statistics(df):
    """FR-3.3: mean, median and 90th percentile of durations using NumPy."""
    heading("9. NUMPY STATISTICS OF CHARGING DURATION  [OUT-4]")
    durations = df["Charging_Duration_Mins"].to_numpy(dtype=float)
    durations = durations[~np.isnan(durations)]
    mean_d = np.mean(durations)
    median_d = np.median(durations)
    p90_d = np.percentile(durations, 90)
    print(f"Mean duration        : {mean_d:.2f} minutes")
    print(f"Median duration      : {median_d:.2f} minutes")
    print(f"90th percentile      : {p90_d:.2f} minutes")
    print("(Mean > median indicates a right-skewed distribution driven by long sessions.)")
    return durations, mean_d, median_d, p90_d