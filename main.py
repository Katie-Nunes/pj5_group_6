import pandas as pd
from numpy import dtype
from datetime import datetime, date, timedelta

TIME_COLS = {"start time", "end time"}
df = pd.read_excel('Excel Files/Bus Planning.xlsx')
timetable = pd.read_excel('Excel Files/Timetable.xlsx')
distancematrix = pd.read_excel('Excel Files/DistanceMatrix.xlsx')

expected_columns = ['start location', 'end location', 'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus']
expected_dtypes = {'start location': dtype('O'), 'end location': dtype('O'), 'start time': dtype('O'), 'end time': dtype('O'), 'activity': dtype('O'), 'line': dtype('float64'), 'energy consumption': dtype('float64'), 'bus': dtype('int64')}

def main (df, expected_columns, expected_dtypes):
    validate_dataframe_structure(df, expected_columns, expected_dtypes)

def get_dtype (df):
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
    print("SUCCESS!")

def check_against_possible_locations(df, timetable, distancematrix):
    df_locations = set(df['start']).union(set(df['end']))
    timetable_locations = set(timetable['start']).union(set(timetable['end']))
    distancematrix_locations = set(distancematrix['start']).union(set(distancematrix['end']))
    all_equal = (df_locations == timetable_locations == distancematrix_locations)

    if all_equal:
        print("All dataframes have matching sets of locations!")
    else:
        print("There's a mismatch in locations between the dataframes.")
    return all_equal

def check_for_innacuracies (df):

    pass
def _coerce(series, ref_date):
    t = pd.to_datetime(series.astype(str), format='%H:%M:%S').dt.time
    return pd.to_datetime([datetime.combine(ref_date, x) for x in t])

def preprocess_planning(df, ref_date=None):
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

def validate_energy_consumption(df):
    df['validation_status'] = 'Valid'  # Initialize all rows as valid
    invalid_rows_found = False

    charging_mask = df['activity'] == 'charging'
    if df.loc[charging_mask, 'energy consumption'] > 0:
        invalid_rows_found = True
        # ASK USER FOR CHOICE
        df.loc[charging_mask, 'energy consumption'] =  df.loc[charging_mask, 'time_taken']*7.5 # Charging per minute = 450/60

    return df

def calculate_insights(df):


    return insights_df

#def check_feasibility ():
#    pass

if __name__ == "__main__":
    main(df, expected_columns, expected_dtypes)
