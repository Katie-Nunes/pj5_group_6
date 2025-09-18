import pandas as pd
# get_dtype, check_for_innacuracies, _coerce, preprocess_planning, check_against_possible_locations
from check_innacuracies import check_for_innacuracies
from numpy import dtype

BUSPLANNING = pd.read_excel('Excel Files/Bus Planning.xlsx')
TIMETABLE = pd.read_excel('Excel Files/Timetable.xlsx')
DISTANCEMATRIX = pd.read_excel('Excel Files/DistanceMatrix.xlsx')
expected_columns = ['start location', 'end location', 'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus']
expected_dtypes = {'start location': dtype('O'), 'end location': dtype('O'), 'start time': dtype('O'), 'end time': dtype('O'), 'activity': dtype('O'), 'line': dtype('float64'), 'energy consumption': dtype('float64'), 'bus': dtype('int64')}

def main (df, expected_columns, expected_dtypes, timetable, distancematrix):
    check_for_innacuracies(df, expected_columns, expected_dtypes, timetable, distancematrix)
    df.to_excel("Excel Files/Changed Planning.xlsx")

if __name__ == "__main__":
    main(BUSPLANNING, expected_columns, expected_dtypes, TIMETABLE, DISTANCEMATRIX)
