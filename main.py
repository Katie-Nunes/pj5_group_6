import pandas as pd
from check_inaccuracies import check_for_inaccuracies, ensure_packages
from check_feasbility import check_feasibility

ensure_packages(['pandas', 'numpy', 'streamlit', 'plotly', 'xlsxwriter', 'datetime'])

PLANNING = pd.read_excel('Excel Files/Bus Planning.xlsx')
TIMETABLE = pd.read_excel('Excel Files/Timetable.xlsx')
DISTANCEMATRIX = pd.read_excel('Excel Files/DistanceMatrix.xlsx')

def main (planning_df, timetable_df, distancematrix_df):
    df = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df)
    check_feasibility(df, TIMETABLE)
    return df

if __name__ == "__main__":
    main(PLANNING, TIMETABLE, DISTANCEMATRIX)
