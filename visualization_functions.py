import streamlit as st
import plotly.express as px
import pandas as pd


def make_gantt(gantt_df):
    fig = px.timeline(gantt_df, x_start="start_dt", x_end="finish_dt", y="bus", color="activity",
                      hover_data=["start location", "end location", "line", "energy consumption"],
                      title="Bus Planning – Daily Gantt")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Time", yaxis_title="", legend_title="Activity",
                     font_size=13, title_font_size=22)
    return fig

def display_insights():
    insights_df = pd.DataFrame({
        "Metric": ["Data Quality", "Rows", "Missing Values", "Exec Time (s)"],
        "Value": ["85 %", "10 520", "45 (0.4 %)", "12.5"],
        "Status": ["✅ OK", "✅ OK", "⚠️ Warn", "✅ OK"]})
    st.dataframe(insights_df, use_container_width=True)

def load_excel_with_fallback(label, key):
    uploaded_file = st.file_uploader(f"{label} (.xlsx)", type=["xlsx"], key=key)

    if uploaded_file:
        return pd.read_excel(uploaded_file)
    return None

def display_df(excel):
    try:
        with st.expander(f"Preview files (first 5 rows)"):
            st.dataframe(excel.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")
