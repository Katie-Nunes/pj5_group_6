from check_inaccuracies import ensure_packages
ensure_packages(['pandas', 'numpy', 'streamlit', 'plotly', 'xlsxwriter', 'datetime'])

import pandas as pd
from check_inaccuracies import check_for_inaccuracies, ensure_packages
from check_feasbility import check_energy_feasibility
from create_planning import create_planning
import streamlit as st

PLANNING = pd.read_excel('Excel Files/Bus Planning.xlsx')
TIMETABLE = pd.read_excel('Excel Files/Timetable.xlsx')
DISTANCEMATRIX = pd.read_excel('Excel Files/DistanceMatrix.xlsx')

def main (planning_df, timetable_df, distancematrix_df, debug=False):
    if not debug:
        try:
            exec(open("app.py").read())
        except Exception as exc:
            st.error("❗ An unexpected error occurred.")
            st.exception(exc)
    else:
        create_planning(TIMETABLE, DISTANCEMATRIX)
        #df = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df)
        #check_energy_feasibility(df, TIMETABLE)

if __name__ == "__main__":
    main(PLANNING, TIMETABLE, DISTANCEMATRIX)
