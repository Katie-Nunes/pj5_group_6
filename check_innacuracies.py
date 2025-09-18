import pandas as pd
from datetime import datetime, date, timedelta
import sys

TIME_COLS = {"start time", "end time"}

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


def check_energy_consumption(df, distancematrix):
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
                low = -7.5 * minutes * 1.1
                high = -7.5 * minutes * 0.9
                if not (low < energy < high):
                    errors.append(f"Row {idx}: Charging energy {energy:.2f} outside range ({low:.2f}, {high:.2f})")

            elif activity in ('material trip', 'service trip'):
                start_loc, end_loc = row['start location'], row['end location']
                trip_key = (start_loc, end_loc)
                distance_m = distance_lookup.get(trip_key)
                if distance_m is None:
                    errors.append(f"Row {idx}: No distance found for trip {start_loc} -> {end_loc}")
                else:
                    distance_km = distance_m / 1000.0
                    low = 0.7 * distance_km
                    high = 2.5 * distance_km
                    if not (low <= energy <= high):
                        errors.append(
                            f"Row {idx}: {activity} energy {energy:.2f} outside range ({low:.2f}, {high:.2f}) for {distance_km:.2f}km")

            elif activity == 'idle':
                low = (5 / 60) * minutes * 0.90
                high = (5 / 60) * minutes * 1.1
                if not (low < energy < high):
                    errors.append(f"Row {idx}: Idle energy {energy:.2f} outside range ({low:.2f}, {high:.2f})")

            else:
                errors.append(f"Row {idx}: Unrecognized activity '{activity}'")

        except Exception as e:
            errors.append(f"Row {idx}: Check error - {str(e)}")

    return errors


def check_datetime_sequence(df):
    """Pure check: Validate datetime continuity and duration."""
    errors = []

    # Check duration validity
    if not (df['finish_dt'] > df['start_dt']).all():
        errors.append("Some rows have negative duration")

    # Check continuity
    continuity = df['start_dt'] == df['finish_dt'].shift(1)
    continuity.iloc[0] = True  # First row always OK
    if not continuity.all():
        errors.append("Datetime sequence has gaps or overlaps")

    return errors


def fill_gaps_with_idle(df):
    """Execution: Fix timeline gaps by inserting idle rows."""
    result_rows = []
    prev_row = None

    for idx, row in df.iterrows():
        if prev_row is not None:
            gap_start = prev_row['finish_dt']
            gap_end = row['start_dt']
            if gap_start < gap_end:
                gap_duration = gap_end - gap_start
                idle_row = prev_row.copy()
                idle_row['start_dt'] = gap_start
                idle_row['finish_dt'] = gap_end
                idle_row['activity'] = 'idle'
                result_rows.append(idle_row)
                # Execution logging (not error)
                print(f"Info: Inserted idle row for {gap_duration} gap between {gap_start} and {gap_end}")
        result_rows.append(row)
        prev_row = row

    return pd.DataFrame(result_rows).reset_index(drop=True)


def fix_charging_energy(df):
    """Execution: Fix charging energy values to standard rate."""
    for idx, row in df.iterrows():
        if row['activity'] == 'charging':
            duration = row['finish_dt'] - row['start_dt']
            minutes = duration.total_seconds() / 60.0
            df.at[idx, 'energy consumption'] = -7.5 * minutes
    return df


def preprocess_planning(df, ref_date=None):
    """Execution: Standardize columns and compute time fields."""
    ref_date = ref_date or date.today()
    df = df.copy()  # Avoid modifying original
    df.columns = df.columns.str.strip().str.lower()

    if missing := TIME_COLS - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    # Coerce time columns to datetime
    df["start_dt"] = _coerce(df["start time"], ref_date)
    df["finish_dt"] = _coerce(df["end time"], ref_date)
    df.loc[df["finish_dt"] < df["start_dt"], "finish_dt"] += timedelta(days=1)
    df["time_taken"] = df["finish_dt"] - df["start_dt"]
    df["bus_str"] = "Bus " + df["bus"].astype(str)

    return df


def _coerce(series, ref_date):
    """Helper: Convert time strings to datetime objects."""
    t = pd.to_datetime(series.astype(str), format='%H:%M:%S').dt.time
    return pd.to_datetime([datetime.combine(ref_date, x) for x in t])


def check_for_innacuracies(df, expected_columns, expected_dtypes, timetable, distancematrix, ref_date=None):
    """Centralized error handling and workflow orchestration."""
    try:
        # Step 1: Validate structure (must pass)
        validate_dataframe_structure(df, expected_columns, expected_dtypes)
    except Exception as e:
        print(f"CRITICAL STRUCTURE ERROR: {e}", file=sys.stderr)
        return df  # Return original df to prevent further processing

    try:
        # Step 2: Preprocess data (execution)
        df = preprocess_planning(df, ref_date)
    except Exception as e:
        print(f"PREPROCESSING ERROR: {e}", file=sys.stderr)
        return df

    # Step 3: Run pure checks and report errors
    location_errors = check_locations(df, timetable, distancematrix, 'ehvgar')
    for err in location_errors:
        print(f"LOCATION ERROR: {err}")

    energy_errors = check_energy_consumption(df, distancematrix)
    for err in energy_errors:
        print(f"ENERGY ERROR: {err}")

    datetime_errors = check_datetime_sequence(df)
    for err in datetime_errors:
        print(f"DATETIME ERROR: {err}")

    # Step 4: Execute fixes (after reporting errors)
    df = fix_charging_energy(df)  # Fix charging energy
    df = fill_gaps_with_idle(df)  # Fix timeline gaps

    return df