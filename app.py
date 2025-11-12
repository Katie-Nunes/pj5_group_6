from check_inaccuracies import ensure_packages
ensure_packages(['pandas', 'numpy', 'streamlit', 'plotly', 'xlsxwriter', 'datetime'])

import pandas as pd
import streamlit as st
import logging
from check_inaccuracies import check_for_inaccuracies, rename_time_object
import app_visualization_functions as avm
from logging_utils import report_error, report_warning, report_info

# Initialize default configuration values in session state (FIX: Initialize before reading)
st.session_state.setdefault('battery_capacity', 300)
st.session_state.setdefault('soh', 0.85)
st.session_state.setdefault('charge_range', (0.1, 0.9))
st.session_state.setdefault('min_charge_minutes', 15)
st.session_state.setdefault('start_end_location', 'ehvgar')

# Now safely read from session state (guaranteed to exist)
full_new_battery = st.session_state.battery_capacity
state_of_health_frac = st.session_state.soh
low, high = st.session_state.charge_range
min_charging_minutes = st.session_state.min_charge_minutes
start_end_location = st.session_state.start_end_location

# --------------------------------------------------------
# Setup
# --------------------------------------------------------
st.set_page_config(page_title="Bus Planning App Dashboard", page_icon="🍆", layout="wide")
st.title("Bus Planning App Dashboard")
st.markdown(
    "📂 Upload your **timetable**, **planning files**, and **distance matrix** on the left. "
    " See insights, diagnostics, and interactive Gantt visualization on the right."
)

# Init logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

left, middle, right = st.columns([4, 10, 3])

# --------------------------------------------------------
# Sidebar / Left Pane: File Upload
# --------------------------------------------------------
with left:
    st.header("File Uploaders")
    planning_df = avm.load_excel_with_fallback("Bus Planning file", "u_planning")
    timetable_df = avm.load_excel_with_fallback("Timetable file", "u_timetable")
    distancematrix_df = avm.load_excel_with_fallback("Distance Matrix", "u_distancematrix")

# --------------------------------------------------------
# Right Pane: Configuration
# --------------------------------------------------------
with right:
    st.header("Variables")
    tab_inn, tab_feas = st.tabs(["Inaccuracy", "Feasibility"])

    with tab_inn:
        st.subheader("Inaccuracy")
        avm.display_inaccuracy_vars()

    with tab_feas:
        st.subheader("Feasibility")
        avm.display_feasibility_vars()

# --------------------------------------------------------
# Middle Pane: Production Tabs
# --------------------------------------------------------
with middle:
    st.header("Production")
    tab_visualize, tab_inspect, tab_insight, tab_optimize = st.tabs(["Visualize", "Inspect", "Insight", "Optimize"])

    # =================================================================
    # Tab 1: Visualize
    # =================================================================
    gantt_df = None  # ensure scope safety
    with tab_visualize:
        st.subheader("Gantt Chart")

        # Basic Gantt
        if planning_df is None:
            st.info("Upload **Bus Planning file** to view basic Gantt chart.")
        else:
            try:
                gantt_df_one = rename_time_object(planning_df, 'start time', 'end time')
                st.toast(
                    f"Loaded {len(gantt_df_one)} trips across "
                    f"{gantt_df_one['bus'].nunique()} bus(es)."
                )
                fig = avm.make_gantt(gantt_df_one)
                st.plotly_chart(fig, use_container_width=True, key="basic_gantt_chart")
            except Exception as e:
                st.error(f"Could not build Gantt chart.\nDetails: {e}")
                logger.exception("Error building basic Gantt")

        # Improved, validated Gantt
        st.subheader("Improved Gantt Chart")
        if not all([planning_df is not None, timetable_df is not None, distancematrix_df is not None]):
            st.info(" Upload **all three files** to view the improved Gantt chart.")
        else:
            try:
                gantt_df = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df)
                
                # Debug: Check activity distribution
                activity_counts = gantt_df['activity'].value_counts()
                logger.info(f"Activity distribution: {activity_counts.to_dict()}")
                
                # Debug: Check for zero-duration rows
                if 'time_taken' in gantt_df.columns:
                    zero_duration = gantt_df[gantt_df['time_taken'] <= pd.Timedelta(0)]
                    if len(zero_duration) > 0:
                        logger.warning(f"Found {len(zero_duration)} rows with zero or negative duration")
                        logger.warning(f"Activities: {zero_duration['activity'].value_counts().to_dict()}")
                
                st.toast(
                    f" ᕙ(  •̀ ᗜ •́  )ᕗ Corrected {len(gantt_df)} trips."
                    f" Covering {gantt_df['bus'].nunique()} bus(es)."
                )
                st.plotly_chart(avm.make_gantt(gantt_df), use_container_width=True, key="improved_gantt_chart")
            except Exception as e:
                st.error(f"( ｡ •̀ ᴖ •́ ｡)  Could not build improved Gantt chart.\nDetails: {e}")
                logger.exception("Error building improved Gantt")

    # =================================================================
    # Tab 2: Inspect
    # =================================================================
    with tab_inspect:
        st.subheader("View Files")

        files = [
            ("Bus Planning", planning_df),
            ("Timetable", timetable_df),
            ("Distance Matrix", distancematrix_df)
        ]
        for name, df in files:
            if df is None:
                st.info(f" Upload {name} file to view.")
            else:
                avm.display_df(df, name)


    # =================================================================
    # Tab 3: Insight
    # =================================================================
with tab_insight:
    try:
        st.subheader("Insights")
        if gantt_df is None or timetable_df is None or distancematrix_df is None:
            st.info("Upload all files for insights/feasibility checks.")
        else:
            try:
                insights_df, feasibility_df = avm.calculate_insights(
                    gantt_df,
                    distancematrix_df,
                    timetable_df,
                    full_new_battery,
                    state_of_health_frac,
                    low, high,
                    min_charging_minutes,
                    start_end_location
                )
            except Exception as e:
                report_error(f"Could not calculate insights: {e}", exception=e)
    except Exception as e:
            st.error(f" Could not calculate insights.\nDetails: {e}")
            import logging
            logging.exception("Error calculating insights")

    # =================================================================
    # Tab 4: Optimize
    # =================================================================
    with tab_optimize:
        if timetable_df is None or distancematrix_df is None:
            st.info("Upload **Timetable** and **Distance Matrix** to run optimization.")
        else:
            try:
                avm.run_packet_optimizer(timetable_df, distancematrix_df)
            except Exception as e:
                st.error(f"Could not run optimizer.\nDetails: {e}")
                logger.exception("Error running packet optimizer")

# --------------------------------------------------------
# Bottom Export
# --------------------------------------------------------
with left:
    st.subheader("Export Bus Planning")
    if gantt_df is None or distancematrix_df is None:
        st.info("Upload all files to enable export.")
    else:
        try:
            avm.export_to_excel(gantt_df)
        except Exception as e:
            st.error(f"Could not export to Excel.\nDetails: {e}")
            logger.exception("Error exporting")
    avm.donate_button()