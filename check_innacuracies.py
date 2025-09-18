import pandas as pd
from datetime import datetime, date, timedelta


TIME_COLS = {"start time", "end time"}

def _get_dtype (df):
    expected_dtypes = dict(df.dtypes)
    return expected_dtypes

def validate_dataframe_structure(df,expected_columns, expected_dtypes):
    actual_columns = df.columns.tolist()
    # Check column names match exactly
    if actual_columns != expected_columns:
        raise ValueError(f"Column names don't match. Expected: {expected_columns.keys}, Got: {actual_columns}")

    # Check each column's dtype
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = df[col].dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"Column '{col}' has wrong dtype. Expected: {expected_dtype}, Got: {actual_dtype}")
    print("Success: Dataframe Structure Validated!")

def check_against_possible_locations(df, timetable, distancematrix):
    df_locations = set(df['start location']).union(set(df['end location']))
    timetable_locations = set(timetable['start']).union(set(timetable['end']))
    distancematrix_locations = set(distancematrix['start']).union(set(distancematrix['end']))

    df_locations.discard('ehvgar') # MAKE THIS GENERAL, currently just removes Garage
    distancematrix_locations.discard('ehvgar')

    all_equal = (df_locations == timetable_locations == distancematrix_locations)

    if all_equal:
        print("Success: All dataframes have matching sets of locations!")
    else:
        print("Error: There's a mismatch in locations between the dataframes. Please fix this!")
    return all_equal

def _coerce(series, ref_date):
    t = pd.to_datetime(series.astype(str), format='%H:%M:%S').dt.time
    return pd.to_datetime([datetime.combine(ref_date, x) for x in t])

def _preprocess_planning(df, ref_date=None):
    ref_date = ref_date or date.today()
    df.columns = df.columns.str.strip().str.lower()
    if missing := TIME_COLS - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    df["start_dt"] = _coerce(df["start time"], ref_date)
    df["finish_dt"] = _coerce(df["end time"], ref_date)
    df.loc[df["finish_dt"] < df["start_dt"], "finish_dt"] += timedelta(days=1)
    df["time_taken"] = df["finish_dt"] - df["start_dt"]
    df["bus_str"] = "Bus " + df["bus"].astype(str)
    return df

import sys

def validate_energy_consumption(df, distancematrix):
    df = _preprocess_planning(df)
    df['validation_status'] = 'Valid'
    
    # Pre-allocate a list so we can give a tidy summary later
    bad_rows = []

    try:
        distance_lookup = distancematrix.set_index(['start', 'end'])['distance_m'].to_dict()
    except KeyError:
        print("FATAL ERROR: The 'distancematrix' DataFrame must contain 'start', 'end', and 'distance m' columns.", file=sys.stderr)
        return df

    for idx, row in df.iterrows():
        try:
            activity = row['activity']
            energy   = row['energy consumption']
            minutes = row['time_taken'].total_seconds() / 60.0

            if activity == 'charging':
                low, high = -7.5 * minutes * 1.1, -7.5 * minutes * 0.9   # both negative
                if not (low < energy < high):
                    bad_rows.append((idx, activity, energy, low, high))
                    df.loc[idx, 'validation_status'] = 'Invalid'
                    df.loc[idx, 'energy consumption'] = -7.5 * minutes

            elif activity in ('material trip', 'service trip'):
                start_loc, end_loc = row['start location'], row['end location']
                trip_key = (start_loc, end_loc)
                distance_m = distance_lookup.get(trip_key)

                distance_km = distance_m / 1000.0
                low  = 0.7 * distance_km
                high = 2.5 * distance_km

                if not (low <= energy <= high):
                    bad_rows.append((idx, activity, energy, low, high))  # Now it's a 5-element tuple
                    df.loc[idx, 'validation_status'] = 'Invalid'

            elif activity == 'idle':
                low, high = (5/60) * minutes * 0.90, (5/60) * minutes * 1.1
                if not (low < energy < high):
                    bad_rows.append((idx, activity, energy, low, high))
                    df.loc[idx, 'validation_status'] = 'Invalid'

            else:
                # completely unknown activity
                raise ValueError(f'unrecognised activity {activity!r}')

        except Exception as err:
            # Any other problem (missing column, NaN, wrong type…)
            print(f'Row {idx}  –  ERROR: {err}', file=sys.stderr)
            df.loc[idx, 'validation_status'] = 'Invalid'
            bad_rows.append((idx, 'ERROR', str(err), None, None))

    if bad_rows:
        print('\nEnergy-consumption violations found:')
        for idx, act, val, lo, hi in bad_rows:
            if act == 'ERROR':
                print(f'  Row {idx:>4}  –  {val}')
            else:
                print(f'  Row {idx:>4}  ({act:13})  '
                      f'energy={val:8.2f}  outside  [{lo:8.2f}, {hi:8.2f}]')
    else:
        print('Success: Energy consumption is roughly accurate for every row!')

    return df

def check_for_innacuracies (df, expected_columns, expected_dtypes, timetable, distancematrix ):
    validate_dataframe_structure(df, expected_columns, expected_dtypes)
    check_against_possible_locations(df, timetable, distancematrix)
    validate_energy_consumption(df, distancematrix)
    return df

def calculate_insights(df):

    pass
    #return insights_df

#def check_feasibility ():
#    pass