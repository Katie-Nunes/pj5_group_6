import streamlit as st
import pandas as pd
import plotly.express as px
import pathlib
from check_innacuracies import _preprocess_planning

# Page config
st.set_page_config(page_title="Bus Planning App Dashboard", page_icon="📥", layout="wide")

def make_gantt(df):
    fig = px.timeline(df, x_start="start_dt", x_end="finish_dt", y="bus_str", color="activity",
                     hover_data=["start location", "end location", "line", "energy consumption"],
                     title="Bus Planning – Daily Gantt")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Time", yaxis_title="", legend_title="Activity", font_size=13, title_font_size=22)
    return fig

# Page body
st.title("Bus Planning App Dashboard")
st.markdown("Upload your timetable and planning files on the left; review insights and the interactive Gantt chart on the right.")
left, right = st.columns([1, 4])

# Left column
with left:
    st.header("File Uploaders")
    timetable_file = st.file_uploader("Timetable file (.xlsx)", type=["xlsx"], key="u_timetable")
    planning_file = st.file_uploader("Bus Planning file (.xlsx)", type=["xlsx"], key="u_planning")

    if timetable_file:
        try:
            tt_df = pd.read_excel(timetable_file) if timetable_file.name.endswith(("xls", "xlsx")) else pd.read_csv(timetable_file)
            st.success(f"Timetable loaded – {tt_df.shape[0]} rows, {tt_df.shape[1]} columns.")
            with st.expander("Preview timetable (first 10 rows)"):
                st.dataframe(tt_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read timetable: {e}")

    st.markdown('<div class="red-button">', unsafe_allow_html=True)

# Right column
with right:
    st.subheader("Gantt Chart")

    planning_df = None
    if planning_file:
        planning_df = pd.read_excel(planning_file)
    elif (fallback := pathlib.Path("Bus Planning.xlsx")).exists():
        planning_df = pd.read_excel(fallback)

    if planning_df is None:
        st.info("Upload a Bus Planning file (.xlsx) to generate the Gantt chart.")
    else:
        try:
            gantt_df = _preprocess_planning(planning_df)
            st.success(f"Loaded {len(gantt_df)} trips for {gantt_df['bus'].nunique()} bus(es).")
            st.plotly_chart(make_gantt(gantt_df), use_container_width=True)
        except Exception as e:
            st.error(f"Could not build Gantt chart: {e}")

    st.markdown("---")
    st.subheader("Insights (Dummy Data for Now)")

    insights_df = pd.DataFrame({
        "Metric": ["Data Quality", "Rows", "Missing Values", "Exec Time (s)"],
        "Value": ["85 %", "10 520", "45 (0.4 %)", "12.5"],
        "Status": ["✅ OK", "✅ OK", "⚠️ Warn", "✅ OK"]})
    st.dataframe(insights_df, use_container_width=True)

# Footer
st.markdown("---")
