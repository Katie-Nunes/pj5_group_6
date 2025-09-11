import pandas as pd
from numpy import dtype

df = pd.read_excel('Excel Files/Bus Planning.xlsx')
timetable = pd.read_excel('Excel Files/Timetable.xlsx')

expected_columns = ['start location', 'end location', 'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus']
expected_dtypes = {'start location': dtype('O'), 'end location': dtype('O'), 'start time': dtype('O'), 'end time': dtype('O'), 'activity': dtype('O'), 'line': dtype('float64'), 'energy consumption': dtype('float64'), 'bus': dtype('int64')}

def main (df, expected_columns, expected_dtypes):
    validate_dataframe_stexpected_dtypesructure(df, expected_columns, expected_dtypes)

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


def check_against_possible_locations ():
    pass

def check_for_innacuracies ():
    pass

#def check_feasibility ():
#    pass

if __name__ == "__main__":
    main(df, expected_columns, expected_dtypes)