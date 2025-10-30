# feasibility_checks.py
import pandas as pd
import streamlit as st

from logging_utils import report_error, report_warning, report_info
from check_inaccuracies import rename_time_object

def energy_state(df, full_new_battery=300, state_of_health_frac=0.85):
    try:
        initial_charge = full_new_battery * state_of_health_frac
        df = df.copy()
        df['cumulative_energy_used'] = df.groupby('bus')['energy consumption'].cumsum()
        df['current_charge'] = initial_charge - df['cumulative_energy_used']
        return df, initial_charge
    except Exception as e:
        report_error("Error computing energy state", e)
        return df, 0


def check_energy_feasibility(df, initial_charge, low=0.1, high=0.9):
    min_bat = initial_charge * low
    max_bat = initial_charge * high
    under = df[df['current_charge'] < min_bat]
    over = df[df['current_charge'] > max_bat]

    if not under.empty:
        failed_indices = under.index.tolist()
        report_error(f"Some buses dip below minimum charge! Failed at rows: {failed_indices}")
        return False
    if not over.empty:
        failed_indices = over.index.tolist()
        report_error(f"Some buses exceed maximum charge threshold! Failed at rows: {failed_indices}")
        return False

    report_info("ᕙ(  •̀ ᗜ •́  )ᕗ All trips are charge feasible", user=True)
    return True

def validate_start_end_locations(df, start_end_location="ehvgar"):
    grouped = df.groupby('bus', as_index=False)
    invalid = pd.DataFrame({
        'bus': grouped.first()['bus'],
        'start': grouped.first()['start location'],
        'end': grouped.last()['end location']
    })
    not_ok = invalid[
        (invalid['start'] != start_end_location) |
        (invalid['end'] != start_end_location)
    ]
    if not not_ok.empty:
        failed_indices = not_ok.index.tolist()
        report_warning(f"Some buses do not start/end at depot {failed_indices}")
    return not_ok

def minimum_charging(df, min_charging_minutes=15):
    charging = df[df['activity'] == 'charging']
    threshold = pd.Timedelta(minutes=min_charging_minutes)
    bad = charging[charging['time_taken'] < threshold]
    if not bad.empty:
        failed_indices = bad.index.tolist()
        report_warning(f"Some charging blocks are shorter than minimum allowed {failed_indices}")
    return bad

def fulfills_timetable(df, timetable_df):
    service_trips = df[df['activity'] == 'service trip']

    service_trip_set = set(zip(
        service_trips['start location'],
        service_trips['end location'],
        service_trips['start time'],
        service_trips['line']
    ))

    timetable_set = set(zip(
        timetable_df['start'],
        timetable_df['end'],
        timetable_df['departure_time'],
        timetable_df['line']
    ))

    missing_trips = timetable_set - service_trip_set

    if missing_trips:
        failed_indices = missing_trips.index.tolist()
        report_error(f"Missing {len(missing_trips)} timetable trips, rows {failed_indices} of timetable are not covered")
        return False, missing_trips

    report_info("ᕙ(  •̀ ᗜ •́  )ᕗ Timetable matches covered")
    return True, set()

def check_all_feasibility(df, timetable_df):
    timetable_df = rename_time_object(timetable_df, "departure_time", "Not Inside")
    df_energy, initial_charge = energy_state(df)
    energy_ok = check_energy_feasibility(df_energy, initial_charge)
    invalid_buses = validate_start_end_locations(df_energy)
    bad_charging = minimum_charging(df_energy)

    timetable_ok, missing_trips = fulfills_timetable(df_energy, timetable_df)

    feasible = energy_ok and timetable_ok

    if not invalid_buses.empty:
        failed_indices = invalid_buses.index.tolist()
        report_warning(f"Found {len(invalid_buses)} buses not starting/ending at depot, at rows {failed_indices}")

    if not bad_charging.empty:
        failed_indices = bad_charging.index.tolist()
        report_warning(f"Found {len(bad_charging)} charging blocks shorter than minimum, at rows {failed_indices}")
    return feasible, missing_trips