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
def _coerce(series: pd.Series, ref_date: date) -> pd.Series:
    """Convert time strings into datetime on a reference date."""
    s = series.astype(str)
    for fmt in ("%H:%M:%S", "%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            t = pd.to_datetime(s, format=fmt).dt.time
            break
        except (ValueError, TypeError):
            continue
    else:
        raise ValueError(f"Invalid time format in series: {s.tolist()[:5]} ...")

    return pd.to_datetime([datetime.combine(ref_date, x) for x in t])


def rename_time_object(df: pd.DataFrame, start_name: str, end_name: str) -> pd.DataFrame:
    """Attach actual datetime objects to time fields."""
    df = df.copy()
    df[start_name] = _coerce(df[start_name], date.today())
    if end_name in df:
        df[end_name] = _coerce(df[end_name], date.today())
    return df


def check_datetime_sequence(df: pd.DataFrame) -> Tuple[List[str], pd.Series]:
    """Check if datetime sequence is continuous and durations valid."""
    errors = []

    if not (df['end time'] > df['start time']).all():
        errors.append("Some rows have negative or zero duration")

    continuity = df['start time'] == df['end time'].shift(1)
    continuity.iloc[0] = True
    if not continuity.all():
        errors.append("Datetime sequence has gaps or overlaps")

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
    """Validate per-row energy consumption against expected physical ranges."""
    errors = []
    distance_lookup = (distancematrix.set_index(['start', 'end'])['distance_m'].to_dict())
    df = df.copy()

    def validate_range(idx, value, low, high, label) -> bool:
        if not low <= value <= high:
            errors.append(f"Row {idx}: {label} energy {value:.2f} outside range [{low:.2f}, {high:.2f}]")
            return False
        return True

    for idx, row in df.iterrows():
        try:
            activity, energy, minutes = row['activity'], row['energy consumption'], row['time_taken'].total_seconds() / 60

            if activity == 'charging':
                low, high = -charge_speed_assumed * minutes * low_charge_rate, -charge_speed_assumed * minutes * high_charge_rate
                if not validate_range(idx, energy, high, low, "Charging"):
                    df.at[idx, 'energy consumption'] = -charge_speed_assumed * minutes

            elif activity in ('material trip', 'service trip'):
                km = distance_lookup.get((row['start location'], row['end location']), None)
                if km is None:
                    errors.append(f"Row {idx}: Unknown trip {row['start location']} → {row['end location']}")
                    continue
                km /= 1000
                low, high = low_energy_use * km, high_energy_use * km
                if not validate_range(idx, energy, low, high, "Trip"):
                    df.at[idx, 'energy consumption'] = km * ((low_energy_use + high_energy_use) / 2)
            elif activity == 'idle':
                base = (idle_cost_ph / 60) * minutes
                low, high = base * low_idle_cost, base * high_idle_cost
                if not validate_range(idx, energy, low, high, "Idle"):
                    df.at[idx, 'energy consumption'] = base

            else:
                errors.append(f"Row {idx}: Unknown activity type {activity}")

        except Exception as e:
            errors.append(f"Row {idx}: Error checking energy - {e}")

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
            report_error(f"Location error: {err}")
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
            report_error(f"Datetime error: {err}")
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
            report_error(f"Energy error: {err}")
    except Exception as e:
        report_error("Energy check failed", e)

    return df