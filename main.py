from check_inaccuracies import ensure_packages
ensure_packages(['pandas', 'numpy', 'streamlit', 'plotly', 'xlsxwriter', 'datetime'])

import pandas as pd
from check_inaccuracies import check_for_inaccuracies, ensure_packages
from check_feasbility import check_all_feasibility, fulfills_timetable
from create_planning import create_planning
import streamlit as st
import logging
for name, l in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        l.disabled = True

PLANNING = pd.read_excel('Excel Files/Bus Planning.xlsx')
TIMETABLE = pd.read_excel('Excel Files/Timetable.xlsx')
DISTANCEMATRIX = pd.read_excel('Excel Files/DistanceMatrix.xlsx')

def main (planning_df, timetable_df, distancematrix_df, debug=True):
    if not debug:
        try:
            exec(open("app.py").read())
        except Exception as exc:
            st.error("❗ An unexpected error occurred.")
            st.exception(exc)
    else:
        #df = create_planning(timetable_df, distancematrix_df)
        #df = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df)
        #tt, missing_trips = check_all_feasibility(df, timetable_df)
        #print(missing_trips)
        df = create_planning(TIMETABLE, DISTANCEMATRIX)
    return df

if __name__ == "__main__":
    df = main(PLANNING, TIMETABLE, DISTANCEMATRIX)