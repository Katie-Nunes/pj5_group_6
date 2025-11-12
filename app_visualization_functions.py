from check_feasbility import (  # assume you put previous functions in feasibility_checks.py
    check_energy_feasibility,
    validate_start_end_locations,
    minimum_charging,
    fulfills_timetable,
    energy_state
)
from check_inaccuracies import rename_time_object
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from logging_utils import report_error


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------
def make_gantt(gantt_df: pd.DataFrame):
    """Generate a Gantt-style timeline chart for buses."""
    fig = px.timeline(
        gantt_df,
        x_start="start time",
        x_end="end time",
        y="bus",
        color="activity",
        color_discrete_map={
            "material trip": 'rgba(131, 201, 255)',
            "service trip": 'rgba(0, 104, 201)',
            "idle": 'rgba(255, 171, 171)',
            "charging": 'red'
        },
        category_orders={"activity": ["material trip", "service trip", "idle", "charging"]},
        hover_data=[
            "start location",
            "end location",
            "line",
            "energy consumption"
        ],
        title="Bus Planning – Daily Gantt"
    )
    fig.update_yaxes(autorange="reversed")  # flip buses descending
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Bus ID",
        legend_title="Activity",
        font=dict(size=20),
        title_font=dict(size=22),

    )
    return fig


# --------------------------------------------------------
# Distance & Insight Calculations
# --------------------------------------------------------
def get_total_distance_km(df: pd.DataFrame, distance_lookup: pd.DataFrame) -> float:
    """Compute total traveled distance in kilometers."""
    merged = df.merge(
        distance_lookup,
        left_on=["start location", "end location"],
        right_on=["start", "end"],
        how="left"
    )
    merged["distance_m"] = merged["distance_m"].fillna(0)
    total_distance_km = merged["distance_m"].sum() / 1000.0
    return total_distance_km

import pandas as pd
import plotly.express as px
import streamlit as st

def _colorize_status(df: pd.DataFrame, col: str = "Status") -> str:
    """Convert DataFrame into HTML table with traffic-light background colors."""
    color_map = {
        "ᕙ(  •̀ ᗜ •́  )ᕗ Good": "background-color:#4CAF50; color:white",
        "ᕙ(  •̀ ᗜ •́  )ᕗ Acceptable": "background-color:#4CAF50; color:white",
        "ᕙ(  •̀ ᗜ •́  )ᕗ Efficient": "background-color:#4CAF50; color:white",
        "ᕙ(  •̀ ᗜ •́  )ᕗ Pass": "background-color:#4CAF50; color:white",
        " (╥ᆺ╥；)  Low": "background-color:#FFEB3B; color:black",
        " (╥ᆺ╥；)  High": "background-color:#FFEB3B; color:black",
        " (╥ᆺ╥；)  Out of range": "background-color:#FF9800; color:black",
        "( ｡ •̀ ᴖ •́ ｡) Fail": "background-color:#F44336; color:white",
    }

    styled = df.copy()
    styles = []
    for _, row in df.iterrows():
        css_row = []
        for colname in df.columns:
            if colname == col:
                css_row.append(color_map.get(str(row[colname]), ""))
            else:
                css_row.append("")
        styles.append(css_row)

    return df.style.set_table_attributes('class="dataframe" style="width:100%; border-collapse:collapse;"') \
                   .set_table_styles([
                        {"selector": "thead th", "props": [("background-color", "#f0f0f0"), ("font-weight", "bold")]}
                   ]) \
                   .apply(lambda _: styles, axis=None)


def calculate_insights(df: pd.DataFrame, distance_lookup: pd.DataFrame,
                       timetable_df: pd.DataFrame,
                       full_new_battery=300,
                       state_of_health_frac=0.85,
                       low=0.1, high=0.9,
                       min_charging_minutes=15,
                       start_end_location="ehvgar") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate and display performance KPIs and feasibility checks with traffic-light coloring.
    """
    # Update parameters from session state if available
    if 'battery_capacity' in st.session_state:
        full_new_battery = st.session_state.battery_capacity
    if 'soh' in st.session_state:
        state_of_health_frac = st.session_state.soh
    if 'charge_range' in st.session_state:
        low, high = st.session_state.charge_range
    if 'min_charge_minutes' in st.session_state:
        min_charging_minutes = st.session_state.min_charge_minutes
    if 'start_end_location' in st.session_state:
        start_end_location = st.session_state.start_end_location


    # --- Core Metrics ---
    n_busses = max(df["bus"])
    total_time = df["time_taken"].sum()
    total_energy = df["energy consumption"].sum()
    activity_sums = df.groupby("activity")["time_taken"].sum()
    total_distance = get_total_distance_km(df, distance_lookup)

    productive_fraction = total_time and (activity_sums.get("service trip", 0) / total_time)
    unproductive_fraction = total_time and (
        (activity_sums.get("idle", 0) + activity_sums.get("material trip", 0)) / total_time
    )
    energy_per_km = total_distance and (total_energy / total_distance)

    # Assign KPI traffic lights
    status_bus = "ᕙ(  •̀ ᗜ •́  )ᕗ Good" if n_busses < 20 else (" (╥ᆺ╥；) A lot of Busses" if n_busses > 25 else "( ｡ •̀ ᴖ •́ ｡) Too Many Busses")
    status_prod = "ᕙ(  •̀ ᗜ •́  )ᕗ Good" if productive_fraction > 0.5 else (" (╥ᆺ╥；)  Low" if productive_fraction > 0.35 else "( ｡ •̀ ᴖ •́ ｡) Fail")
    status_unp = "ᕙ(  •̀ ᗜ •́  )ᕗ Good" if unproductive_fraction < 0.5 else (" (╥ᆺ╥；)  High" if unproductive_fraction < 0.65 else "( ｡ •̀ ᴖ •́ ｡) Fail")
    status_epkm = "ᕙ(  •̀ ᗜ •́  )ᕗ Good" if 0 < energy_per_km < 3 else (" (╥ᆺ╥；)  Out of range" if 0 < energy_per_km < 5 else "( ｡ •̀ ᴖ •́ ｡) Fail")

    insights_df = pd.DataFrame({
        "Metric": [
            "Number of Busses",
            "Productive Time Fraction (%)",
            "Unproductive Time Fraction (%)",
            "Energy Use (kWh / service km)"
        ],
        "Value": [
            float(round(n_busses, 2)) if productive_fraction else 0,
            float(round(productive_fraction * 100, 2)) if productive_fraction else 0,
            float(round(unproductive_fraction * 100, 2)) if unproductive_fraction else 0,
            float(round(energy_per_km, 2)) if energy_per_km else 0
        ],
        "Status": [status_bus, status_prod, status_unp, status_epkm]
    })

    st.markdown("### Performance KPIs")
    st.dataframe(insights_df.style.map(
        lambda val: "background-color:#4CAF50; color:white" if "ᕙ(  •̀ ᗜ •́  )ᕗ" in str(val)
        else ("background-color:#FFEB3B; color:black" if " (╥ᆺ╥；) " in str(val)
              else ("background-color:#F44336; color:white" if "( ｡ •̀ ᴖ •́ ｡)" in str(val) else "")),
        subset=["Status"]
    ), use_container_width=True)

    # Pie chart
    pie_df = pd.DataFrame({
        "Activity": activity_sums.index,
        "Time (hours)": activity_sums.dt.total_seconds() / 3600
    })
    st.plotly_chart(px.pie(
        pie_df, names="Activity", values="Time (hours)",
        title="Total Time Distribution", hole=0.3,
        color="Activity",
        color_discrete_map={
            "material trip": 'rgba(131, 201, 255)',
            "service trip": 'rgba(0, 104, 201)',
            "idle": 'rgba(255, 171, 171)',
            "charging": 'red'
        },
        category_orders={"activity": ["material trip", "service trip", "idle", "charging"]},),
        use_container_width=True,
        key="insight_pie_chart"
    )

    st.markdown("### Feasibility Checks")
    df_soc, initial_charge = energy_state(df, full_new_battery, state_of_health_frac)

    feas_data = []



    soc_ok = check_energy_feasibility(df_soc, initial_charge, low, high)
    feas_data.append({"Check": "Battery charge within bounds", "Result": "ᕙ(  •̀ ᗜ •́  )ᕗ Pass" if soc_ok else "( ｡ •̀ ᴖ •́ ｡) Fail"})

    invalid_buses = validate_start_end_locations(df_soc, start_end_location)
    feas_data.append({"Check": f"Depot start/end at {start_end_location}",
                      "Result": "ᕙ(  •̀ ᗜ •́  )ᕗ Pass" if invalid_buses.empty else f"( ｡ •̀ ᴖ •́ ｡) {len(invalid_buses)} fail"})

    insufficient = minimum_charging(df_soc, min_charging_minutes)
    feas_data.append({"Check": f"Min charging ≥ {min_charging_minutes} min",
                      "Result": "ᕙ(  •̀ ᗜ •́  )ᕗ Pass" if insufficient.empty else f"( ｡ •̀ ᴖ •́ ｡) {len(insufficient)} short"})

    timetable_df = rename_time_object(timetable_df, "departure_time", "Not Inside")
    is_valid, mismatched = fulfills_timetable(df_soc, timetable_df)
    feas_data.append({"Check": "Timetable coverage",
                      "Result": "ᕙ(  •̀ ᗜ •́  )ᕗ Pass" if is_valid else f"( ｡ •̀ ᴖ •́ ｡) {len(mismatched)} unmatched"})

    feasibility_df = pd.DataFrame(feas_data)

    st.dataframe(feasibility_df.style.map(
        lambda val: "background-color:#4CAF50; color:white" if "ᕙ(  •̀ ᗜ •́  )ᕗ" in str(val)
        else ("background-color:#F44336; color:white" if "( ｡ •̀ ᴖ •́ ｡)" in str(val)
              else ("background-color:#FFEB3B; color:black" if " (╥ᆺ╥；) " in str(val) else "")),
        subset=["Result"]
    ), use_container_width=True)

    return insights_df, feasibility_df


# --------------------------------------------------------
# Excel Utilities
# --------------------------------------------------------

def load_excel_with_fallback(label: str, key: str) -> pd.DataFrame | None:
    """Upload an Excel spreadsheet into a DataFrame."""
    uploaded_file = st.file_uploader(f"{label} (.xlsx)", type=["xlsx"], key=key)
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading {label}: {e}")
            return None
    return None

def export_to_excel(df: pd.DataFrame, filename: str = "gantt_export.xlsx"):
    """Download DataFrame as Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Gantt Data", index=False)
    output.seek(0)

    st.download_button(
        label="Download Excel",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# --------------------------------------------------------
# Controls
# --------------------------------------------------------
import streamlit as st

def display_feasibility_vars():
    # Capture values and store in session state
    st.session_state.battery_capacity = st.number_input(
        "Full New Battery (kWh)", 0, 1000, 300, step=10,
        help="Nominal new-battery capacity"
    )
    st.session_state.soh = st.slider(
        "State of Health Fraction", 0.0, 1.0, 0.85, 0.01,
        help="Battery degradation factor"
    )
    st.session_state.charge_range = st.slider(
        "Charge Feasibility Range", 0.0, 1.0, (0.1, 0.9), 0.01,
        help="Allowed min/max SOC fraction"
    )
    st.session_state.min_charge_minutes = st.number_input(
        "Minimum Charging Minutes", 0, 240, 15, step=1
    )
    st.session_state.start_end_location = st.text_input(
        "Must Start/End Location", value="ehvgar"
    )

def display_inaccuracy_vars():
    st.session_state.max_idle = st.number_input(
        "Max Idle Period (minutes)", 0, 1440, 120, 1
    )
    st.session_state.idle_cost = st.number_input(
        "Idle Cost (kWh/hr)", 0.0, 100.0, 5.0, 0.1
    )
    st.session_state.charge_speed = st.number_input(
        "Charge Speed (kW)", 0.0, 100.0, 7.5, 0.1
    )
    st.session_state.charge_error = st.slider(
        "Charge Error Margin", 0.0, 2.0, (0.9, 1.1), 0.01
    )
    st.session_state.energy_margin = st.slider(
        "Energy Use Margin (kWh/km)", 0.0, 5.0, (0.7, 2.5), 0.01
    )
    st.session_state.idle_margin = st.slider(
        "Idle Cost Margin", 0.0, 2.0, (0.9, 1.1), 0.01
    )
    st.session_state.discard_location = st.text_input(
        "Discard Location", "ehvgar"
    )

def display_df(excel: pd.DataFrame, label: str = "Files"):
    """Browse DataFrame in collapsible preview."""
    try:
        with st.expander(f"Preview {label} (first 5 rows)"):
            st.dataframe(excel.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not preview {label}: {e}")


# --------------------------------------------------------
# Packet Optimizer
# --------------------------------------------------------
def run_packet_optimizer(timetable_df: pd.DataFrame, distance_matrix_df: pd.DataFrame):
    """Run the single-packet bus optimizer and display results."""
    from packet_optimizer import optimize_bus_planning, OptimizerConfig
    
    st.subheader("Packet-Based Bus Optimization")
    st.markdown("""
    This optimizer uses a **single circular packet** of buses that rotate through trips sequentially.
    When a bus needs charging, it temporarily exits the rotation and the next bus continues.
    This minimizes the total fleet size while ensuring all trips are covered.
    """)
    
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Battery Settings**")
            battery_capacity = st.number_input(
                "Battery Capacity (kWh)", 
                min_value=100.0, 
                max_value=500.0, 
                value=300.0, 
                step=10.0,
                help="Total battery capacity when new"
            )
            soh = st.slider(
                "State of Health (%)", 
                min_value=50, 
                max_value=100, 
                value=85, 
                step=5,
                help="Battery degradation factor"
            ) / 100.0
            min_soc = st.slider(
                "Min SOC (%)", 
                min_value=5, 
                max_value=50, 
                value=10, 
                step=5,
                help="Minimum allowed state of charge"
            ) / 100.0
            max_soc = st.slider(
                "Max SOC (%)", 
                min_value=50, 
                max_value=100, 
                value=90, 
                step=5,
                help="Maximum allowed state of charge"
            ) / 100.0
        
        with col2:
            st.markdown("**Charging & Energy**")
            charging_rate = st.number_input(
                "Charging Rate (kWh/min)", 
                min_value=1.0, 
                max_value=20.0, 
                value=7.5, 
                step=0.5,
                help="Rate at which battery charges"
            )
            charging_window = st.number_input(
                "Charging Window (minutes)", 
                min_value=10, 
                max_value=60, 
                value=15, 
                step=5,
                help="Duration of each charging session"
            )
            energy_per_km = st.number_input(
                "Energy per km (kWh/km)", 
                min_value=0.5, 
                max_value=3.0, 
                value=1.5, 
                step=0.1,
                help="Energy consumption per kilometer"
            )
            idle_energy = st.number_input(
                "Idle Energy (kWh/min)", 
                min_value=0.01, 
                max_value=0.5, 
                value=0.05, 
                step=0.01,
                help="Energy consumed while idling (not at garage)"
            )
        
        with col3:
            st.markdown("**Advanced Settings**")
            garage_location = st.text_input(
                "Garage Location", 
                value="ehvgar",
                help="Location code for the depot/garage"
            )
            charging_buffer = st.slider(
                "Charging Buffer (%)", 
                min_value=5, 
                max_value=30, 
                value=15, 
                step=5,
                help="Safety margin: send to charge when SOC drops to (min + buffer)%"
            ) / 100.0
    
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Optimizing fleet size..."):
            config = OptimizerConfig(
                battery_capacity=battery_capacity,
                state_of_health=soh,
                min_soc_percent=min_soc,
                max_soc_percent=max_soc,
                charging_rate_kwh_per_min=charging_rate,
                charging_window_minutes=int(charging_window),
                energy_per_km=energy_per_km,
                idle_energy_per_min=idle_energy,
                garage_location=garage_location,
                charging_buffer_percent=charging_buffer
            )
            
            try:
                optimized_df, stats = optimize_bus_planning(
                    timetable_df, 
                    distance_matrix_df, 
                    config
                )
                
                if not optimized_df.empty:
                    st.success(f"✅ Optimization Complete! Minimum fleet size: **{stats['fleet_size']} buses**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Fleet Size", stats['fleet_size'])
                    with col2:
                        st.metric("Service Trips", stats['total_service_trips'])
                    with col3:
                        st.metric("Charging Sessions", stats['total_charging_sessions'])
                    with col4:
                        st.metric("Total Distance (km)", f"{stats['total_distance_km']:.1f}")
                    
                    st.subheader("Optimized Schedule Gantt Chart")
                    fig = make_gantt(optimized_df)
                    st.plotly_chart(fig, use_container_width=True, key="optimized_gantt_chart")
                    
                    st.subheader("Detailed Statistics")
                    stats_df = pd.DataFrame([{
                        'Metric': k.replace('_', ' ').title(),
                        'Value': f"{v:.2f}" if isinstance(v, float) else str(v)
                    } for k, v in stats.items()])
                    st.dataframe(stats_df, use_container_width=True)
                    
                    st.subheader("Download Optimized Planning")
                    excel_buffer = BytesIO()
                    optimized_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_buffer,
                        file_name="optimized_bus_planning.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.session_state['optimized_planning'] = optimized_df
                else:
                    st.error("Optimization failed to produce a valid schedule.")
            
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                import traceback
                st.code(traceback.format_exc())


# --------------------------------------------------------
# Donation
# --------------------------------------------------------
@st.dialog("Donate")
def donate_popup():
    st.markdown("### ☕ Buy me a Coffee?")
    st.markdown("""
    - **Bank Transfer**: [Rabobank Link](https://betaalverzoek.rabobank.nl/betaalverzoek/?id=0g-XTRZfTcmkVAh08KRo3Q)
    - **Crypto (XMR)**: [AnonPay](https://trocador.app/anonpay?ticker_to=xmr&network_to=Mainnet&address=82jERfWaYWMLaZRjgwvZ5TgJCVoSfRSFRNp9oyPhAsso1nMjqvyZ1sxgguy4NLnmCiV8C4S4tFegYZCGKn6CChbYUUJE5bm&ref=sqKNYGZbRl&direct=True)
    """)


def donate_button():
    """Small footer donation button."""
    st.button("Donate", type="primary", on_click=donate_popup)