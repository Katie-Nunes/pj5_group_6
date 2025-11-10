import pandas as pd
import numpy as np
import logging
import importlib
import subprocess
import sys
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple
from logging_utils import report_error, report_warning, report_info


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# -----------------------------
# Utility
# -----------------------------
def ensure_packages(packages: List[str]) -> None:
    """Ensure given packages are installed, install if missing."""
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            logger.warning(f"Package '{package}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# -----------------------------
# Validation & Structure Checks
# -----------------------------


def validate_dataframe_structure(
    df: pd.DataFrame,
    expected_dtypes: dict = None,
    strict_order: bool = False,
    apply: bool = False
) -> bool:
    """Check DF matches schema. Returns True if ok, False if not."""
    expected_dtypes = expected_dtypes or {
        'start location': np.object_,
        'end location': np.object_,
        'start time': np.object_,
        'end time': np.object_,
        'activity': np.object_,
        'line': np.floating,
        'energy consumption': np.floating,
        'bus': np.integer
    }

    if apply:
        # Convert dtypes safely
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                try:
                    if dtype == 'Int64':
                        # Use nullable integer type for 'bus' column
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    elif dtype in ['float64', np.floating]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' to {dtype}: {e}")
        return df

    ok = True
    expected_cols = list(expected_dtypes.keys())
    actual_cols = df.columns.tolist()

    if strict_order and actual_cols != expected_cols:
        report_error(f"Column order mismatch! Expected {expected_cols}, got {actual_cols}")
        ok = False
    else:
        missing = [c for c in expected_cols if c not in actual_cols]
        extra = [c for c in actual_cols if c not in expected_cols]
        if missing:
            report_error(f"Missing columns: {missing}")
            ok = False
        if extra:
            report_warning(f"Ignoring unexpected columns: {extra}")

    for col, expected_dtype in expected_dtypes.items():
        if col in df and not np.issubdtype(df[col].dtype, expected_dtype):
            report_error(
                f"Column '{col}' wrong dtype → Expected {expected_dtype}, got {df[col].dtype}"
            )
            ok = False
    return ok

def check_locations(
        df: pd.DataFrame, timetable: pd.DataFrame, distancematrix: pd.DataFrame, discard: str
) -> List[str]:
    """Check location consistency across input dataframes."""
    df_locations = set(df['start location']).union(df['end location'])
    timetable_locations = set(
        timetable.get('start', timetable['start'])
    ).union(timetable.get('end', timetable['end']))
    distancematrix_locations = set(distancematrix['start']).union(distancematrix['end'])

    for locations in (df_locations, distancematrix_locations):
        locations.discard(discard)

    if not (df_locations == timetable_locations == distancematrix_locations):
        return ["Location mismatch across dataframes."]
    return []


# -----------------------------
# Time Handling
# -----------------------------
import pandas as pd
from datetime import date, datetime


def _coerce(series: pd.Series, ref_date: date) -> pd.Series:
    """Convert time strings into datetime on a reference date, handling invalid values."""
    # Ensure the input is a string series to handle mixed types (e.g., numbers, NaN)
    s = series.astype(str)

    t = None  # Initialize t to None
    for fmt in ("%H:%M:%S", "%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            # Attempt to parse the entire series with the current format
            parsed_datetime = pd.to_datetime(s, format=fmt, errors='coerce')

            # Check if any values were successfully parsed
            if parsed_datetime.notna().any():
                t = parsed_datetime.dt.time
                print(f"  - Successfully used format '{fmt}' to parse times.")
                break  # Exit loop on the first successful format
        except Exception as e:
            # This catch is less likely to be hit now with errors='coerce',
            # but kept for robustness.
            print(f"  - Format '{fmt}' failed with error: {e}")
            continue

    if t is None:
        # This block runs if no format could parse any value
        raise ValueError(f"Could not parse any time values in series. Found formats: {s.unique().tolist()}")

    # THE FIX: Handle NaT values before combining with the date
    final_datetimes = [datetime.combine(ref_date, x) if not pd.isna(x) else pd.NaT for x in t]

    return pd.to_datetime(final_datetimes)


def rename_time_object(df: pd.DataFrame, start_name: str, end_name: str) -> pd.DataFrame:
    """Attach actual datetime objects to time fields."""
    df = df.copy()
    df[start_name] = _coerce(df[start_name], date.today())
    if end_name in df:
        df[end_name] = _coerce(df[end_name], date.today())
    return df


def check_datetime_sequence(df: pd.DataFrame) -> Tuple[List[str], pd.Series]:
    errors = []

    # Ensure columns are datetime for robust comparison
    df['start time'] = pd.to_datetime(df['start time'])
    df['end time'] = pd.to_datetime(df['end time'])

    # 1. Check for negative or zero duration
    # Find rows where end time is not after start time
    invalid_duration_mask = df['end time'] <= df['start time']
    if invalid_duration_mask.any():
        # Get the index of rows with invalid duration
        invalid_indices = df[invalid_duration_mask].index.tolist()
        errors.append(
            f"Negative or zero duration found at indices: {invalid_indices}"
        )

    # 2. Check for gaps or overlaps in the sequence
    # A sequence is continuous if the start time of the current row
    # equals the end time of the previous row.
    continuity = df['start time'] == df['end time'].shift(1)

    # The first row has no previous row, so it's considered continuous by definition.
    if not continuity.empty:
        continuity.iloc[0] = True

    if not continuity.all():
        # Get the index of rows where continuity is False
        # These are the rows that start the gap or overlap
        gap_overlap_indices = continuity[~continuity].index.tolist()
        errors.append(
            f"Gaps or overlaps found starting at indices: {gap_overlap_indices}"
        )

    return errors, continuity


# -----------------------------
# Gap Manipulation
# -----------------------------
def insert_idle_given_row(df: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    """Insert an idle segment to fill a gap."""
    if row_idx >= len(df) - 1:
        return df  # no next row

    row_to_copy = df.iloc[row_idx].copy()
    next_row = df.iloc[row_idx + 1]

    row_to_copy.update({
        "start location": row_to_copy["end location"],
        "end location": next_row["start location"],
        "start time": row_to_copy["end time"],
        "end time": next_row["start time"],
        "activity": "idle",
        "line": np.nan,
    })

    return pd.concat([
        df.iloc[: row_idx + 1],
        pd.DataFrame([row_to_copy], columns=df.columns),
        df.iloc[row_idx + 1:]
    ], ignore_index=True)


def fill_all_gaps(df: pd.DataFrame, continuity: pd.Series) -> pd.DataFrame:
    """Fill all identified gaps in dataframe with 'idle' rows."""
    for idx in sorted(continuity[~continuity].index, reverse=True):
        df = insert_idle_given_row(df, idx - 1)
    return df


# -----------------------------
# Processing Helpers
# -----------------------------
def remove_wrong_gaps(df: pd.DataFrame, too_long_for_idle_in_minutes: int = 120) -> pd.DataFrame:
    """Drop idle rows if gaps are too large to be realistic idle time."""
    threshold = pd.Timedelta(minutes=too_long_for_idle_in_minutes)
    return df[df['time_taken'] < threshold]


def rename_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN line IDs with default, and cast to int."""
    df = df.copy()
    df["line"] = df["line"].fillna(999).astype(int)
    return df


def calc_timedelta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duration from start/end times."""
    df = df.copy()
    df.loc[df["end time"] < df["start time"], "end time"] += timedelta(days=1)
    df["time_taken"] = df["end time"] - df["start time"]
    return df


# -----------------------------
# Energy Validation
# -----------------------------

def check_energy_consumption(
        df: pd.DataFrame,
        distancematrix: pd.DataFrame,
        idle_cost_ph: float,
        charge_speed_assumed: float,
        low_charge_rate: float,
        high_charge_rate: float,
        low_energy_use: float,
        high_energy_use: float,
        low_idle_cost: float,
        high_idle_cost: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate per-row energy consumption against expected physical ranges, summarizing all row indices per error type."""
    errors = []
    df = df.copy()
    distance_lookup = distancematrix.set_index(['start', 'end'])['distance_m'].to_dict()

    # Collect row indices by error category
    invalid_charging, invalid_trip, invalid_idle = [], [], []
    unknown_trip, unknown_activity = [], []
    generic_errors = []

    for idx, row in df.iterrows():
        try:
            activity = row['activity']
            energy = row['energy consumption']
            minutes = row['time_taken'].total_seconds() / 60

            if activity == 'charging':
                low = -charge_speed_assumed * minutes * low_charge_rate
                high = -charge_speed_assumed * minutes * high_charge_rate
                if not (high <= energy <= low):
                    invalid_charging.append(idx)
                    df.at[idx, 'energy consumption'] = -charge_speed_assumed * minutes

            elif activity in ('material trip', 'service trip'):
                km = distance_lookup.get((row['start location'], row['end location']))
                if km is None:
                    unknown_trip.append(idx)
                    continue
                km /= 1000
                low = low_energy_use * km
                high = high_energy_use * km
                if not (low <= energy <= high):
                    invalid_trip.append(idx)
                    df.at[idx, 'energy consumption'] = km * ((low_energy_use + high_energy_use) / 2)

            elif activity == 'idle':
                base = (idle_cost_ph / 60) * minutes
                low = base * low_idle_cost
                high = base * high_idle_cost
                if not (low <= energy <= high):
                    invalid_idle.append(idx)
                    df.at[idx, 'energy consumption'] = base

            else:
                unknown_activity.append(idx)

        except Exception as e:
            generic_errors.append((idx, str(e)))

    # Summarize all errors in compact messages
    if invalid_charging:
        errors.append(f"Charging energy outside expected range at rows {invalid_charging}, this is fixed automatically")
    if invalid_trip:
        errors.append(f"Trip energy outside expected range at rows {invalid_trip}, this is fixed automatically")
    if invalid_idle:
        errors.append(f"Idle energy outside expected range at rows {invalid_idle}, this is fixed automatically")
    if unknown_trip:
        errors.append(f"Unknown trip start/end combination at rows {unknown_trip}")
    if unknown_activity:
        errors.append(f"Unknown activity type at rows {unknown_activity}")
    if generic_errors:
        formatted = [f"Row {i}: {msg}" for i, msg in generic_errors]
        errors.append("Unexpected validation errors:\n" + "\n".join(formatted))

    return df, errors


# -----------------------------
# Central Workflow
# -----------------------------
# check_inaccuracies.py
def check_for_inaccuracies(df, timetable, distancematrix,
                           too_long_for_idle_in_minutes=120,
                           idle_cost_ph=5,
                           charge_speed_assumed=7.5,
                           low_charge_rate=0.9,
                           high_charge_rate=1.1,
                           low_energy_use=0.7,
                           high_energy_use=2.5,
                           low_idle_cost=0.9,
                           high_idle_cost=1.1,
                           discard='ehvgar'):
    """
    Main orchestration: catch + log errors/warnings, return corrected DF.
    """
    # Schema
    if not validate_dataframe_structure(df):
        report_error("Critical: Schema invalid, downstream steps may fail.")

    # Location consistency
    try:
        location_errors = check_locations(df, timetable, distancematrix, discard)
        for err in location_errors:
            report_warning(f"Location error: {err}")
    except Exception as e:
        report_error("Error validating locations", e)

    # Parse datetimes
    try:
        df = rename_time_object(df, 'start time', 'end time')
    except Exception as e:
        report_error("Error parsing time columns", e)

    # Sequence check
    try:
        datetime_errors, continuity = check_datetime_sequence(df)
        for err in datetime_errors:
            report_warning(f"Datetime error: {err}, this is fixed automatically")
    except Exception as e:
        report_error("Error checking datetime sequence", e)
        continuity = pd.Series([True] * len(df))

    try:
        df = fill_all_gaps(df, continuity)
        df = calc_timedelta(df)
        df = remove_wrong_gaps(df, too_long_for_idle_in_minutes)
        df = rename_lines(df)
    except Exception as e:
        report_error("Error fixing gaps/lines", e)

    # Energy constraints
    try:
        df, energy_errors = check_energy_consumption(
            df, distancematrix,
            idle_cost_ph, charge_speed_assumed,
            low_charge_rate, high_charge_rate,
            low_energy_use, high_energy_use,
            low_idle_cost, high_idle_cost
        )
        for err in energy_errors:
            report_warning(f"Energy error: {err}")
    except Exception as e:
        report_error("Energy check failed", e)

    return df