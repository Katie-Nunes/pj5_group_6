from check_inaccuracies import ensure_packages
ensure_packages(['pandas', 'numpy', 'streamlit', 'plotly', 'xlsxwriter', 'datetime'])

import pandas as pd
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
        final_planning = pd.read_excel('Excel Files/IIImprovedBusPlanning.xlsx')
        final_planning['current energy'] = 255 - (final_planning.groupby('bus')['energy consumption'].cumsum())

        print("\nSaving results...")
        final_planning.to_excel("Excel Files/IIIImprovedBusPlanning.xlsx", index=False)

        #df = create_planning(timetable_df, distancematrix_df)
        #df = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df)
        #tt, missing_trips = check_all_feasibility(df, timetable_df)
    return None

if __name__ == "__main__":
    df = main(PLANNING, TIMETABLE, DISTANCEMATRIX)