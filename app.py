import streamlit as st
import pandas as pd
import plotly.express as px
import pathlib
from check_innacuracies import check_for_innacuracies

# Page config
st.set_page_config(page_title="Bus Planning App Dashboard", page_icon="📥", layout="wide")

def make_gantt(dataframe):
    fig = px.timeline(dataframe, x_start="start_dt", x_end="finish_dt", y="bus", color="activity",
                      hover_data=["start location", "end location", "line", "energy consumption"],
                      title="Bus Planning – Daily Gantt")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Time", yaxis_title="", legend_title="Activity", font_size=13, title_font_size=22)
    return fig

# Page body
st.title("Bus Planning App Dashboard")
st.markdown("Upload your timetable and planning files on the left; review insights and the interactive Gantt chart on the right.")
left, right = st.columns([1, 4])

def display_df(excel):
    if excel:
        try:
            tt_df = pd.read_excel(excel) if excel.name.endswith(("xls", "xlsx")) else pd.read_csv(excel)
            with st.expander(f"Preview files (first 5 rows)"):
                st.dataframe(tt_df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read timetable: {e}")


# Left column
with left:
    st.header("File Uploaders")
    timetable_file = st.file_uploader("Timetable file (.xlsx)", type=["xlsx"], key="u_timetable")
    planning_file = st.file_uploader("Bus Planning file (.xlsx)", type=["xlsx"], key="u_planning")
    distancematrix_file = st.file_uploader("Distance Matrix (.xlsx)", type=["xlsx"], key="u_distancematrix")

    st.markdown('<div class="red-button">', unsafe_allow_html=True)

# Right column
with (right):
    st.subheader("Gantt Chart")

    planning_df = None
    timetable_df = None
    distancematrix_df = None

    if planning_file:
        planning_df = pd.read_excel(planning_file)
    elif (fallback := pathlib.Path("Excel Files/Bus Planning.xlsx")).exists():
        planning_df = pd.read_excel(fallback)

    if timetable_file:
        timetable_df = pd.read_excel(timetable_file)
    elif (fallback := pathlib.Path("Excel Files/Timetable.xlsx")).exists():
        timetable_df = pd.read_excel(fallback)

    if distancematrix_file:
        distancematrix_df = pd.read_excel(planning_file)
    elif (fallback := pathlib.Path("Excel Files/DistanceMatrix.xlsx")).exists():
        distancematrix_df = pd.read_excel(fallback)

    if planning_df is None or timetable_df is None or distancematrix_df is None:
        st.info("One or more input DataFrames are missing or empty. Upload both Timetable and Bus Planning files (.xlsx) to generate the Gantt chart.")
    else:
        try:
            expected_columns = ['start location', 'end location',  'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus', ]
            expected_dtypes = {'start location': 'object', 'end location': 'object', 'start time': 'object', 'end time': 'object', 'activity': 'object', 'line': 'float64', 'energy consumption': 'float64','bus': 'int64', }
            df = check_for_innacuracies(planning_df, expected_columns, expected_dtypes, timetable_df, distancematrix_df)

            # Check if df is None before proceeding
            if df is None:
                st.error("No valid data returned from check_for_innacuracies().")
            else:
                st.success(f"Loaded {len(df)} trips for {df['bus'].nunique()} bus(es).")
                fig = make_gantt(df)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not build Gantt chart: {e}")

    st.markdown("---")
    st.subheader("Insights (Dummy Data for Now)")

    insights_df = pd.DataFrame({
        "Metric": ["Data Quality", "Rows", "Missing Values", "Exec Time (s)"],
        "Value": ["85 %", "10 520", "45 (0.4 %)", "12.5"],
        "Status": ["✅ OK", "✅ OK", "⚠️ Warn", "✅ OK"]})
    st.dataframe(insights_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Data")

    display_df(timetable_file)
    display_df(planning_file)
    display_df(distancematrix_file)

# Footer
st.markdown("---")