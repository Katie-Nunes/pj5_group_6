import streamlit as st
from check_inaccuracies import check_for_inaccuracies, rename_time_object
from visualization_functions import make_gantt, display_df, calculate_insights, load_excel_with_fallback
from numpy import dtype

st.set_page_config(page_title="Bus Planning App Dashboard", page_icon="📥", layout="wide")
st.title("Bus Planning App Dashboard")
st.markdown("Upload your timetable and planning files on the left; review insights and the interactive Gantt chart on the right.")
left, right = st.columns([1, 4])

with left:
    st.header("File Uploaders")
    planning_df = load_excel_with_fallback("Bus Planning file", "u_planning")
    timetable_df = load_excel_with_fallback("Timetable file","u_timetable")
    distancematrix_df = load_excel_with_fallback("Distance Matrix","u_distancematrix")

with (right):
    st.header("Production")
    st.subheader("Gantt Chart")

    if planning_df is None:
        st.info("Upload all Bus Planning to view Gantt chart.")
    else:
        try:
            gantt_df_one = rename_time_object(planning_df)
            st.success(f"Loaded {len(gantt_df_one)} trips for {gantt_df_one['bus'].nunique()} bus(es).")
            fig = make_gantt(gantt_df_one)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not build Gantt chart: {e}")

## "Improved" Gantt chart, runs check for innacuracies
    if planning_df is None or timetable_df is None or distancematrix_df is None:
        st.info("Upload all files to view Improved Gantt chart.")
    else:
        try:
            expected_columns = ['start location', 'end location', 'start time', 'end time', 'activity', 'line',
                                'energy consumption', 'bus']
            expected_dtypes = {'start location': dtype('O'), 'end location': dtype('O'), 'start time': dtype('datetime64[ns]'),
                               'end time': dtype('datetime64[ns]'), 'activity': dtype('O'), 'line': dtype('float64'),
                               'energy consumption': dtype('float64'), 'bus': dtype('int64')}
            gantt_df = check_for_inaccuracies(planning_df, expected_columns, expected_dtypes, timetable_df, distancematrix_df)
            st.success(f"Loaded {len(gantt_df)} trips for {gantt_df['bus'].nunique()} bus(es).")
            st.plotly_chart(make_gantt(gantt_df), use_container_width=True)
        except Exception as e:
            st.error(f"Could not build Gantt chart: {e}")

    st.subheader("Insights")

    if planning_df is None or timetable_df is None or distancematrix_df is None:
        st.info("Upload all files to view Insights.")
    else:
        calculate_insights(gantt_df)

    st.subheader("View Files")

    if planning_df is None or timetable_df is None or distancematrix_df is None:
        st.info("Upload all files to view Files.")
    else:
        display_df(planning_df)
        display_df(timetable_df)
        display_df(distancematrix_df)