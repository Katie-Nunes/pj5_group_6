import streamlit as st
import plotly.express as px
import pandas as pd

def make_gantt(gantt_df):
    fig = px.timeline(gantt_df, x_start="start time", x_end="end time", y="bus", color="activity",
                      hover_data=["start location", "end location", "line", "energy consumption"],
                      title="Bus Planning – Daily Gantt")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Time", yaxis_title="", legend_title="Activity",
                     font_size=13, title_font_size=22)
    return fig


def calculate_insights(df):
    # Calculate total time and energy consumption
    total_time = df['time_taken'].sum()
    total_energy = df['energy consumption'].sum()

    # Group by activity and sum time_taken
    activity_sums = df.groupby('activity')['time_taken'].sum()

    total_service = activity_sums.get('service trip', 0)
    total_material = activity_sums.get('material trip', 0)
    total_idle = activity_sums.get('idle', 0)

    pd_f = total_service / total_time if total_time != 0 else 0
    unp_f = (total_idle + total_material) / total_time if total_time != 0 else 0

    data = {
        'Metric': ['Productive Time Fraction', 'Unproductive Time Fraction', 'Energy Use'],
        'Value': [pd_f, unp_f, total_energy],
        'Status': ['?', '?', '?']
    }
    insights_df = pd.DataFrame(data=data)

    st.dataframe(insights_df, use_container_width=True)
    return insights_df

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
