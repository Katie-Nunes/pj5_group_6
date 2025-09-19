import pandas as pd
from numpy import dtype

PLANNING = pd.read_excel('Excel Files/Bus Planning.xlsx')
TIMETABLE = pd.read_excel('Excel Files/Timetable.xlsx')
DISTANCEMATRIX = pd.read_excel('Excel Files/DistanceMatrix.xlsx')

expected_columns = ['start location', 'end location', 'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus']
expected_dtypes = {'start location': dtype('O'), 'end location': dtype('O'), 'start time': dtype('O'), 'end time': dtype('O'), 'activity': dtype('O'), 'line': dtype('float64'), 'energy consumption': dtype('float64'), 'bus': dtype('int64')}

def main (df, expected_columns, expected_dtypes, timetable, distancematrix):
    exec(open("app.py").read())

if __name__ == "__main__":
    main(PLANNING, expected_columns, expected_dtypes, TIMETABLE, DISTANCEMATRIX)
