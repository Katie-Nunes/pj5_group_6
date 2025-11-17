from check_inaccuracies import ensure_packages
ensure_packages(['pandas', 'numpy', 'streamlit', 'plotly', 'xlsxwriter', 'datetime'])

import pandas as pd
import streamlit as st
import logging
from check_inaccuracies import check_for_inaccuracies, rename_time_object
import app_visualization_functions as avm
from logging_utils import report_error, report_warning, report_info

# Initialize default configuration values in session state
st.session_state.setdefault('battery_capacity', 300)
st.session_state.setdefault('soh', 0.85)
st.session_state.setdefault('charge_range', (0.1, 0.9))
st.session_state.setdefault('min_charge_minutes', 15)
st.session_state.setdefault('start_end_location', 'ehvgar')

# Page configuration
st.set_page_config(
    page_title="Bus Planning & Optimization",
    page_icon=None,
    layout="wide"
)

# Store optimizer results in session state
if 'optimizer_gantt_df' not in st.session_state:
    st.session_state.optimizer_gantt_df = None

# Debug: Show session state status in sidebar
if st.session_state.optimizer_gantt_df is not None:
    with st.sidebar:
        st.success(f"Optimized data available: {len(st.session_state.optimizer_gantt_df)} trips")

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================================================================
# SIDEBAR: File Uploads & Configuration
# ============================================================================
with st.sidebar:
    st.title("Bus Planner")
    
    st.header("Data Files")
    planning_df = avm.load_excel_with_fallback("Bus Planning", "u_planning")
    timetable_df = avm.load_excel_with_fallback("Timetable", "u_timetable")
    distancematrix_df = avm.load_excel_with_fallback("Distance Matrix", "u_distancematrix")
    
    st.divider()
    
    st.header("Configuration")
    
    with st.expander("Validation Settings", expanded=False):
        avm.display_inaccuracy_vars()
    
    with st.expander("Feasibility Settings", expanded=False):
        avm.display_feasibility_vars()
    
    st.divider()
    
    st.header("Export")
    if planning_df is None or timetable_df is None or distancematrix_df is None:
        st.info("Upload all files to enable export")
    else:
        gantt_df_temp = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df) if all([planning_df is not None, timetable_df is not None, distancematrix_df is not None]) else None
        if gantt_df_temp is not None:
            avm.export_to_excel(gantt_df_temp)
    
    avm.donate_button()

# ============================================================================
# Pre-process data before tabs (so it's available to all tabs)
# ============================================================================
# Process original schedule
gantt_df_original = None
if planning_df is not None:
    gantt_df_original = rename_time_object(planning_df, 'start time', 'end time')
    # Add time_taken column if missing
    if 'time_taken' not in gantt_df_original.columns:
        if 'start time' in gantt_df_original.columns and 'end time' in gantt_df_original.columns:
            gantt_df_original['time_taken'] = gantt_df_original['end time'] - gantt_df_original['start time']

# Process validated schedule
gantt_df_validated = None
if all([planning_df is not None, timetable_df is not None, distancematrix_df is not None]):
    try:
        gantt_df_validated = check_for_inaccuracies(planning_df, timetable_df, distancematrix_df)
    except Exception as e:
        logger.exception("Error creating validated schedule")

# ============================================================================
# MAIN AREA: Content Tabs
# ============================================================================
st.title("Bus Planning & Optimization Dashboard")
st.markdown("Analyze bus schedules, validate feasibility, and optimize fleet size")

tab_schedules, tab_performance, tab_data, tab_optimize = st.tabs([
    "Schedules", 
    "Performance", 
    "Data", 
    "Optimize"
])

# =================================================================
# Tab 1: Schedules
# =================================================================
with tab_schedules:
    st.subheader("Original")
    if gantt_df_original is None:
        st.info("Upload Bus Planning file to view original schedule")
    else:
        st.write(f"Loaded {len(gantt_df_original)} trips across {gantt_df_original['bus'].nunique()} bus(es)")
        fig = avm.make_gantt(gantt_df_original)
        st.plotly_chart(fig, use_container_width=True, key="original_gantt_chart")
    
    st.divider()
    
    st.subheader("Validated")
    if gantt_df_validated is None:
        st.info("Upload all three files for validated schedule")
    else:
        st.write(f"Validated {len(gantt_df_validated)} trips across {gantt_df_validated['bus'].nunique()} bus(es)")
        st.plotly_chart(avm.make_gantt(gantt_df_validated), use_container_width=True, key="validated_gantt_chart")
    
    st.divider()
    
    st.subheader("Optimized")
    if st.session_state.optimizer_gantt_df is not None:
        opt_df = st.session_state.optimizer_gantt_df
        st.write(f"Optimized {len(opt_df)} trips across {opt_df['bus'].nunique()} bus(es)")
        st.plotly_chart(avm.make_gantt(opt_df), use_container_width=True, key="optimized_gantt_chart")
    else:
        st.info("Run optimizer to view optimized schedule")

# =================================================================
# Tab 2: Performance
# =================================================================
with tab_performance:
    full_new_battery = st.session_state.battery_capacity
    state_of_health_frac = st.session_state.soh
    low, high = st.session_state.charge_range
    min_charging_minutes = st.session_state.min_charge_minutes
    start_end_location = st.session_state.start_end_location
    
    st.subheader("Original")
    if gantt_df_original is None or distancematrix_df is None or timetable_df is None:
        st.info("Upload all files to view original performance")
    else:
        try:
            avm.calculate_insights(gantt_df_original, distancematrix_df, timetable_df,
                                 full_new_battery, state_of_health_frac, low, high,
                                 min_charging_minutes, start_end_location)
        except Exception as e:
            st.error(f"Could not calculate original performance: {e}")
            logger.exception("Error calculating original performance")
    
    st.divider()
    
    st.subheader("Validated")
    if gantt_df_validated is None or timetable_df is None or distancematrix_df is None:
        st.info("Upload all files to view validated performance")
    else:
        try:
            avm.calculate_insights(gantt_df_validated, distancematrix_df, timetable_df,
                                 full_new_battery, state_of_health_frac, low, high,
                                 min_charging_minutes, start_end_location)
        except Exception as e:
            st.error(f"Could not calculate validated performance: {e}")
            logger.exception("Error calculating validated performance")
    
    st.divider()
    
    st.subheader("Optimized")
    if st.session_state.optimizer_gantt_df is not None and distancematrix_df is not None and timetable_df is not None:
        try:
            avm.calculate_insights(st.session_state.optimizer_gantt_df, distancematrix_df, timetable_df,
                                 full_new_battery, state_of_health_frac, low, high,
                                 min_charging_minutes, start_end_location)
        except Exception as e:
            st.error(f"Could not calculate optimized performance: {e}")
            logger.exception("Error calculating optimized performance")
    else:
        st.info("Run optimizer and upload all files to view optimized performance")

# =================================================================
# Tab 3: Data
# =================================================================
with tab_data:
    st.subheader("Original")
    if gantt_df_original is None:
        st.info("Upload Bus Planning file to view original data")
    else:
        avm.display_df(gantt_df_original, "Bus Planning (Original)")
    
    st.divider()
    
    st.subheader("Validated")
    if gantt_df_validated is None:
        st.info("Upload all files to view validated data")
    else:
        avm.display_df(gantt_df_validated, "Bus Planning (Validated)")
    
    st.divider()
    
    st.subheader("Optimized")
    if st.session_state.optimizer_gantt_df is not None:
        avm.display_df(st.session_state.optimizer_gantt_df, "Bus Planning (Optimized)")
    else:
        st.info("Run optimizer to view optimized data")
    
    st.divider()
    
    st.subheader("Reference Data")
    if timetable_df is not None:
        avm.display_df(timetable_df, "Timetable")
    else:
        st.info("Upload Timetable file")
    
    if distancematrix_df is not None:
        avm.display_df(distancematrix_df, "Distance Matrix")
    else:
        st.info("Upload Distance Matrix file")

# =================================================================
# Tab 4: Optimize
# =================================================================
with tab_optimize:
    if timetable_df is None or distancematrix_df is None:
        st.info("Upload Timetable and Distance Matrix to run optimization")
    else:
        try:
            # Optimizer updates session state internally
            avm.run_packet_optimizer(timetable_df, distancematrix_df)
        except Exception as e:
            st.error(f"Could not run optimizer: {e}")
            logger.exception("Error running packet optimizer")