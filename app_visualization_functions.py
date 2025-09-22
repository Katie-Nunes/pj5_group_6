import streamlit as st
import plotly.express as px
import pandas as pd
from io import BytesIO
from error_handling import streamlit_error_handler

@streamlit_error_handler
def make_gantt(gantt_df):
    fig = px.timeline(gantt_df, x_start="start time", x_end="end time", y="bus", color="activity",
                      hover_data=["start location", "end location", "line", "energy consumption"],
                      title="Bus Planning – Daily Gantt")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Time", yaxis_title="", legend_title="Activity",
                     font_size=13, title_font_size=22)
    return fig

@streamlit_error_handler
def get_total_distance_km(df, distance_lookup):
    total_distance = 0.0
    for idx, row in df.iterrows():
        start = row['start location']
        end = row['end location']
        if start == end:
            distance_m = 0
        else:
            # Find the matching row in distance_lookup
            match = distance_lookup[
                (distance_lookup['start'] == start) &
                (distance_lookup['end'] == end)
            ]
            if not match.empty:
                distance_m = match.iloc[0]['distance_m']
            else:
                distance_m = 0  # or handle missing distance as you prefer
        total_distance += distance_m
    return total_distance / 1000.0  # returns distance in kilometers

@streamlit_error_handler
def calculate_insights(df, distance_lookup):
    # Calculate total time and energy consumption
    total_time = df['time_taken'].sum()
    total_energy = df['energy consumption'].sum()

    # Group by activity and sum time_taken
    activity_sums = df.groupby('activity')['time_taken'].sum()

    total_service = activity_sums.get('service trip', 0)
    total_material = activity_sums.get('material trip', 0)
    total_idle = activity_sums.get('idle', 0)
    total_distance = get_total_distance_km(df, distance_lookup)

    pd_f = total_service / total_time if total_time != 0 else 0
    unp_f = (total_idle + total_material) / total_time if total_time != 0 else 0
    epkm = total_energy / total_distance if total_distance != 0 else 0

    data = {
        'Metric': ['Productive Time Fraction', 'Unproductive Time Fraction', 'Energy Use per Service Km'],
        'Value': [pd_f, unp_f, epkm],
        'Status': ['?', '?', '?']
    }
    insights_df = pd.DataFrame(data=data)

    st.dataframe(insights_df, use_container_width=True)

    # --- Plotly Pie Chart of Total Time per Activity ---
    # Convert time to minutes for better readability
    activity_time_hours = activity_sums.dt.total_seconds() / 60 / 60 # convert to hours

    pie_df = pd.DataFrame({
        'Activity': activity_sums.index,
        'Time (hours)': activity_time_hours
    })

    fig = px.pie(
        pie_df,
        names='Activity',
        values='Time (hours)',
        title='Total Time per Activity (hours)',
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

    return insights_df

@streamlit_error_handler
def load_excel_with_fallback(label, key):
    uploaded_file = st.file_uploader(f"{label} (.xlsx)", type=["xlsx"], key=key)
    if uploaded_file:
        return pd.read_excel(uploaded_file)
    return None

@streamlit_error_handler
def export_to_excel(df, filename="gantt_export.xlsx"):
    """Convert DataFrame to Excel bytes for download"""
    output = BytesIO()

    # Create Excel writer object
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Gantt Data', index=False)

    # Reset buffer position and return bytes
    output.seek(0)
    excel_data = output.getvalue()
    st.download_button(
        label="Download Excel File",
        data=excel_data,
        file_name="gantt_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    return

@streamlit_error_handler
def display_feasibility_vars():
    full_new_battery = st.number_input("Full New Battery (kWh)", min_value=0, max_value=1000, value=300, step=10)
    state_of_health_frac = st.slider("State of Health Fraction", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    low, high = st.slider("Charge Feasibility Range", min_value=0.0, max_value=1.0, value=(0.1, 0.9), step=0.01)
    min_charging_minutes = st.number_input("Minimum Charging Minutes", min_value=0, max_value=240, value=15, step=1)
    start_end_location = st.text_input("Must Start and End Location", value='ehvgar')

def display_inaccuracy_vars():
    too_long_for_idle_in_minutes = st.number_input("Too Long for Idle (minutes)", min_value=0, max_value=1440,value=120, step=1)
    idle_cost_ph = st.number_input("Idle Cost Assumed & Assigned (kWh)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    charge_speed_assumed = st.number_input("Charge Speed Assumed & Assigned (kWh)", min_value=0.0, max_value=100.0, value=7.5,step=0.1)
    low_charge_rate, high_charge_rate = st.slider("Charge Margin of Error(kWh)", min_value=0.0, max_value=2.0, value=(0.9, 1.1),step=0.01)
    low_energy_use, high_energy_use = st.slider("Energy Use Margin of Error (kWh/km)", min_value=0.0, max_value=5.0,value=(0.7, 2.5), step=0.01)
    low_idle_cost, high_idle_cost = st.slider("Idle Cost Margin of Error (kWh)", min_value=0.0, max_value=2.0, value=(0.9, 1.1),step=0.01)
    discard = st.text_input("Discard in Location Check", value='ehvgar')


@streamlit_error_handler
def display_df(excel, label="Files"):
    with st.expander(f"Preview {label} (first 5 rows)"):
        st.dataframe(excel.head(5), use_container_width=True)

@st.dialog("💸 Donate")
def donate_popup():
    st.markdown("### Buy me a Coffee?")
    st.markdown(""" 
        - Bank Transfer: https://betaalverzoek.rabobank.nl/betaalverzoek/?id=0g-XTRZfTcmkVAh08KRo3Q
        - Crypto: https://trocador.app/anonpay?ticker_to=xmr&network_to=Mainnet&address=82jERfWaYWMLaZRjgwvZ5TgJCVoSfRSFRNp9oyPhAsso1nMjqvyZ1sxgguy4NLnmCiV8C4S4tFegYZCGKn6CChbYUUJE5bm&ref=sqKNYGZbRl&direct=True 
        """)
    st.markdown("---")

def donate_button():
    if st.button("💸 Donate", type="primary"):
        donate_popup()
