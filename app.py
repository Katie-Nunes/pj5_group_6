import streamlit as st
from check_inaccuracies import check_for_inaccuracies, rename_time_object
import app_visualization_functions as avm
from numpy import dtype

st.set_page_config(page_title="Bus Planning App Dashboard", page_icon="🍆", layout="wide")
st.title("Bus Planning App Dashboard")
st.markdown("Upload your timetable and planning files on the left; review insights and the interactive Gantt chart on the right.")
left, middle, right = st.columns([3, 10, 2])

with left:
    st.header("File Uploaders")
    planning_df = avm.load_excel_with_fallback("Bus Planning file", "u_planning")
    timetable_df = avm.load_excel_with_fallback("Timetable file", "u_timetable")
    distancematrix_df = avm.load_excel_with_fallback("Distance Matrix", "u_distancematrix")

with right:
    st.header("Variables")
    tab_inn, tab_feas = st.tabs(["Inaccuracy", "Feasibility"])
    with tab_inn:
        st.subheader("Inaccuracy")
        avm.display_inaccuracy_vars()
    with tab_feas:
        st.subheader("Feasibility")
        avm.display_feasibility_vars()

with middle:
    st.header("Production")
    tab_visualize, tab_inspect, tab_insight = st.tabs(["Visualize", "Inspect", "Insight"])

    # =================================================================
    # Tab 1: Visualize
    # =================================================================
    with tab_visualize:
        st.subheader("Gantt Chart")
        if planning_df is None:
            st.info("Upload Bus Planning file to view Gantt chart.")
        else:
            try:
                gantt_df_one = rename_time_object(planning_df, 'start time', 'end time')
                st.toast(f"Loaded {len(gantt_df_one)} trips for {gantt_df_one['bus'].nunique()} bus(es).")
                fig = avm.make_gantt(gantt_df_one)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not build Gantt chart: {e}")

        st.subheader("Improved Gantt Chart")
        if planning_df is None or timetable_df is None or distancematrix_df is None:
            st.info("Upload all files to view Improved Gantt chart.")
        else:
            try:
                gantt_df = check_for_inaccuracies(planning_df, timetable_df,distancematrix_df)
                st.toast(f"Loaded {len(gantt_df)} trips for {gantt_df['bus'].nunique()} bus(es).")
                st.plotly_chart(avm.make_gantt(gantt_df), use_container_width=True)
            except Exception as e:
                st.error(f"Could not build Gantt chart: {e}")

    # =================================================================
    # Tab 2: Inspect
    # =================================================================
    with tab_inspect:
        st.subheader("View Files")
        if planning_df is None:
            st.info("Upload Bus Planning file to view.")
        else:
            avm.display_df(planning_df, "Bus Planning")

        if timetable_df is None:
            st.info("Upload Timetable file to view.")
        else:
            avm.display_df(timetable_df, "Timetable")

        if distancematrix_df is None:
            st.info("Upload Distance Matrix file to view.")
        else:
            avm.display_df(distancematrix_df, "Distance Matrix")

    # =================================================================
    # Tab 3: Insight
    # =================================================================
    with tab_insight:
        st.subheader("Insights")
        if planning_df is None or timetable_df is None or distancematrix_df is None:
            st.info("Upload all files to view Insights.")
        else:
            avm.calculate_insights(gantt_df, distancematrix_df)

with left:
    st.subheader("Export Bus Planning")
    if planning_df is None or timetable_df is None or distancematrix_df is None:
        st.info("Upload all files to be able to export to excel.")
    else:
        avm.export_to_excel(gantt_df)
    avm.donate_button()