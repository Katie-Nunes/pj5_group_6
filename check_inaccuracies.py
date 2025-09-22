import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from numpy import dtype
import importlib
import subprocess
import sys

def ensure_packages(packages):
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Package '{package}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def validate_dataframe_structure(df, expected_columns, expected_dtypes):
    """Pure check: Validate column names and dtypes without modifying data."""
    actual_columns = df.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(f"Column names don't match. Expected: {expected_columns}, Got: {actual_columns}")

    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = df[col].dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"Column '{col}' has wrong dtype. Expected: {expected_dtype}, Got: {actual_dtype}")

def check_locations(df, timetable, distancematrix, discard):
    """Pure check: Verify location consistency across dataframes."""
    df_locations = set(df['start location']).union(set(df['end location']))
    try:
        timetable_locations = set(timetable['start']).union(set(timetable['end']))
    except:
        timetable_locations = set(timetable['start location']).union(set(timetable['end']))
    distancematrix_locations = set(distancematrix['start']).union(set(distancematrix['end']))

    df_locations.discard(discard)
    distancematrix_locations.discard(discard)

    if not (df_locations == timetable_locations == distancematrix_locations):
        return ["Location mismatch: Dataframes have inconsistent location sets"]
    return []

def _coerce(series, ref_date):
    """Helper: Convert time strings to datetime objects."""
    try:
        t = pd.to_datetime(series.astype(str), format='%H:%M:%S').dt.time
    except:
        t = pd.to_datetime(series.astype(str), format='%H:%M').dt.time
    return pd.to_datetime([datetime.combine(ref_date, x) for x in t])

def rename_time_object(df, start_name, end_name):
    """Execution: Compute time fields."""
    df[start_name] = _coerce(df[start_name], date.today())
    try:
        df[end_name] = _coerce(df[end_name], date.today())
    except:
        pass
    return df

def check_datetime_sequence(df):
    """Pure check: Validate datetime continuity and duration."""
    errors = []

    # Check duration validity
    if not (df['end time'] > df['start time']).all():
        # ASK USER INPUT
        df = df[df['end time'] > df['start time']]
        errors.append("Some rows have negative duration")

    continuity = df['start time'] == df['end time'].shift(1) # Check if start is always the same as previous end (finds gaps)
    continuity.iloc[0] = True  # First row always OK
    if not continuity.all():
        errors.append("Datetime sequence has gaps or overlaps")
        # Good place to ask user for input

    return errors, continuity

def insert_idle_given_row(df: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    row_to_copy = df.iloc[row_idx].copy()
    next_row = df.iloc[row_idx +1]

    row_to_copy["start location"] = row_to_copy["end location"]
    row_to_copy["end location"] = next_row["start location"]

    row_to_copy["start time"] = row_to_copy["end time"]
    row_to_copy["end time"] = next_row["start time"]

    row_to_copy['activity'] = 'idle'
    row_to_copy['line'] = np.nan

    top    = df.iloc[:row_idx +1]          # inclusive of row_idx (the original row we just duplicated)
    bottom = df.iloc[row_idx +1:]          # from the original next row onward

    new_row_df = pd.DataFrame([row_to_copy], columns=df.columns)
    new_df = pd.concat([top, new_row_df, bottom], ignore_index=True)
    return new_df

def fill_all_gaps(df: pd.DataFrame, continuity) -> pd.DataFrame:
    indices_to_insert = continuity[~continuity].index

    # Sort indices in descending order to avoid index shifting issues
    for idx in sorted(indices_to_insert, reverse=True):
        df = insert_idle_given_row(df, idx-1)
    return df

def remove_wrong_gaps(df, too_long_for_idle_in_minutes=120):
    threshold = pd.Timedelta(seconds=too_long_for_idle_in_minutes*60)
    df = df[df['time_taken'] < threshold]
    return df

def rename_lines(df):
    df["line"] = df["line"].fillna("999")
    df["line"] = df["line"].astype(int)
    return df

def calc_timedelta(df):
    df.loc[df["end time"] < df["start time"], "end time"] += timedelta(days=1)
    df["time_taken"] = df["end time"] - df["start time"]
    return df

def check_energy_consumption(df, distancematrix, idle_cost_ph, charge_speed_assumed: float, low_charge_rate: float, high_charge_rate: float, low_energy_use, high_energy_use, low_idle_cost, high_idle_cost):
    """Check: Validate energy consumption rules without modifying data."""
    errors = []
    distance_lookup = (distancematrix.set_index(['start', 'end'])['distance_m'].to_dict())
    df = df.copy()

    def validate_range(idx, value, low, high, label):
        if not (low <= value <= high):
            errors.append(f"Row {idx}: {label} energy {value:.2f} outside range ({low:.2f}, {high:.2f})")
            return False
        return True

    for idx, row in df.iterrows():
        try:
            activity = row['activity']
            energy = row['energy consumption']
            minutes = row['time_taken'].total_seconds() / 60.0

            if activity == 'charging':
                low = -charge_speed_assumed * minutes * low_charge_rate
                high = -charge_speed_assumed * minutes * high_charge_rate
                if not validate_range(idx, energy, high, low, "Charging"):
                    df.loc[idx, 'energy consumption'] = -charge_speed_assumed * minutes

            elif activity in ('material trip', 'service trip'):
                trip_key = (row['start location'], row['end location'])
                distance_m = distance_lookup.get(trip_key)
                km = distance_m / 1000.0
                low, high = low_energy_use * km, high_energy_use * km
                if not validate_range(idx, energy, low, high, activity):
                    df.loc[idx, 'energy consumption'] = km * ((low_energy_use+high_energy_use)/2)

            elif activity == 'idle':
                base = (idle_cost_ph / 60) * minutes
                low, high = base * low_idle_cost, base * high_idle_cost
                if not validate_range(idx, energy, low, high, "Idle"):
                    df.loc[idx, 'energy consumption'] = (idle_cost_ph/60) * minutes

            else:
                errors.append(f"Row {idx}: Unrecognized activity '{activity}'")

        except Exception as e:
            errors.append(f"Row {idx}: Check error - {e}")

    return df, errors


def check_for_inaccuracies(df, timetable, distancematrix, expected_columns=['start location', 'end location', 'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus'], expected_dtypes={'start location': dtype('O'), 'end location': dtype('O'), 'start time': dtype('O'), 'end time': dtype('O'), 'activity': dtype('O'), 'line': dtype('float64'), 'energy consumption': dtype('float64'), 'bus': dtype('int64')}, too_long_for_idle_in_minutes=120, idle_cost_ph=5, charge_speed_assumed=7.5, low_charge_rate=0.9, high_charge_rate=1.1, low_energy_use=0.7, high_energy_use=2.5, low_idle_cost=0.9, high_idle_cost=1.1, discard='ehvgar', ref_date=None):
    """Centralized error handling and workflow orchestration."""
    try:
        pass
        validate_dataframe_structure(df, expected_columns, expected_dtypes)
    except Exception as e:
        print(f"CRITICAL STRUCTURE ERROR: {e}", file=sys.stderr)

    location_errors = check_locations(df, timetable, distancematrix, discard)
    for err in location_errors:
        print(f"LOCATION ERROR: {err}")

    try:
        df = rename_time_object(df, 'start time', 'end time')
    except Exception as e:
        print(f"PREPROCESSING ERROR: {e}", file=sys.stderr)

    datetime_errors, cunty = check_datetime_sequence(df)
    for err in datetime_errors:
        print(f"DATETIME ERROR: {err}")

    df = fill_all_gaps(df, cunty)
    df = calc_timedelta(df)
    df = remove_wrong_gaps(df, too_long_for_idle_in_minutes)
    df = rename_lines(df)

    df, energy_errors = check_energy_consumption(df, distancematrix, idle_cost_ph, charge_speed_assumed, low_charge_rate, high_charge_rate, low_energy_use, high_energy_use, low_idle_cost, high_idle_cost)
    for err in energy_errors:
        print(f"ENERGY ERROR: {err}")
    return df
