import pandas as pd
from datetime import datetime, date, timedelta
import sys
import numpy as np

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
    timetable_locations = set(timetable['start']).union(set(timetable['end']))
    distancematrix_locations = set(distancematrix['start']).union(set(distancematrix['end']))

    df_locations.discard(discard)
    distancematrix_locations.discard(discard)

    if not (df_locations == timetable_locations == distancematrix_locations):
        return ["Location mismatch: Dataframes have inconsistent location sets"]
    return []

def _coerce(series, ref_date):
    """Helper: Convert time strings to datetime objects."""
    t = pd.to_datetime(series.astype(str), format='%H:%M:%S').dt.time
    return pd.to_datetime([datetime.combine(ref_date, x) for x in t])

def rename_time(df):
    """Execution: Compute time fields."""
    df["start time"] = _coerce(df["start time"], date.today())
    df["end time"] = _coerce(df["end time"], date.today())
    return df

def check_datetime_sequence(df):
    """Pure check: Validate datetime continuity and duration."""
    errors = []

    # Check duration validity
    if not (df['end time'] > df['start time']).all():
        errors.append("Some rows have negative duration")

    continuity = df['start time'] == df['end time'].shift(1)
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

def check_energy_consumption(df, distancematrix, idle_cost_ph, charge_speed: float, low_charge_rate: float, high_charge_rate: float, low_energy_use, high_energy_use, low_idle_cost, high_idle_cost):
    """Pure check: Validate energy consumption rules without modifying data."""
    errors = []

    try:
        distance_lookup = distancematrix.set_index(['start', 'end'])['distance_m'].to_dict()
    except KeyError:
        return ["FATAL: distancematrix missing required columns (start, end, distance_m)"]

    for idx, row in df.iterrows():
        try:
            activity = row['activity']
            energy = row['energy consumption']
            minutes = row['time_taken'].total_seconds() / 60.0

            if activity == 'charging':
                low = -charge_speed * minutes * low_charge_rate
                high = -charge_speed * minutes * high_charge_rate
                if not (high < energy < low):
                    errors.append(f"Row {idx}: Charging energy {energy:.2f} outside range ({low:.2f}, {high:.2f})")

            elif activity in ('material trip', 'service trip'):
                start_loc, end_loc = row['start location'], row['end location']
                trip_key = (start_loc, end_loc)
                distance_m = distance_lookup.get(trip_key)
                if distance_m is None:
                    errors.append(f"Row {idx}: No distance found for trip {start_loc} -> {end_loc}")
                else:
                    distance_km = distance_m / 1000.0
                    low = low_energy_use * distance_km
                    high = high_energy_use * distance_km
                    if not (low <= energy <= high):
                        errors.append(
                            f"Row {idx}: {activity} energy {energy:.2f} outside range ({low:.2f}, {high:.2f}) for {distance_km:.2f}km")

            elif activity == 'idle':
                low = (idle_cost_ph / 60) * minutes * low_idle_cost
                high = (idle_cost_ph / 60) * minutes * high_idle_cost
                if not (low < energy < high):
                    errors.append(f"Row {idx}: Idle energy {energy:.2f} outside range ({low:.2f}, {high:.2f})")

            else:
                errors.append(f"Row {idx}: Unrecognized activity '{activity}'")

        except Exception as e:
            errors.append(f"Row {idx}: Check error - {str(e)}")

    return errors

def fix_charging_energy(df):
    """Execution: Fix charging energy values to standard rate."""
    for idx, row in df.iterrows():
        if row['activity'] == 'charging':
            duration = row['end time'] - row['start time']
            minutes = duration.total_seconds() / 60.0
            df.at[idx, 'energy consumption'] = -7.5 * minutes
    return df

def rename_lines(df):
    df["line"] = df["line"].fillna("999")
    return df

def calc_timedelta(df):
    df.loc[df["end time"] < df["start time"], "end time"] += timedelta(days=1)
    df["time_taken"] = df["end time"] - df["start time"]
    return df

def check_for_inaccuracies(df, expected_columns, expected_dtypes, timetable, distancematrix, ref_date=None):
    """Centralized error handling and workflow orchestration."""
    try:
        pass
        validate_dataframe_structure(df, expected_columns, expected_dtypes)
    except Exception as e:
        print(f"CRITICAL STRUCTURE ERROR: {e}", file=sys.stderr)

    location_errors = check_locations(df, timetable, distancematrix, 'ehvgar')
    for err in location_errors:
        print(f"LOCATION ERROR: {err}")

    try:
        df = rename_time(df)
    except Exception as e:
        print(f"PREPROCESSING ERROR: {e}", file=sys.stderr)

    datetime_errors, cunty = check_datetime_sequence(df)
    for err in datetime_errors:
        print(f"DATETIME ERROR: {err}")

    df = fill_all_gaps(df, cunty)

    energy_errors = check_energy_consumption(df, distancematrix, 5, 7.5, 0.9, 1.1, 0.7, 2.5, 0.9, 1.1 )
    for err in energy_errors:
        print(f"ENERGY ERROR: {err}")

    df = fix_charging_energy(df)
    df = rename_lines(df)
    df = calc_timedelta(df)
    return df
