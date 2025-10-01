# feasibility_checks.py
import pandas as pd
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
        report_error("Some buses dip below minimum charge!")
        return False
    if not over.empty:
        report_error("Some buses exceed maximum charge threshold")
        return False

    report_info("✅ All trips are charge feasible", user=True)
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
        report_warning("Some buses do not start/end at depot")
    return not_ok

def minimum_charging(df, min_charging_minutes=15):
    charging = df[df['activity'] == 'charging']
    threshold = pd.Timedelta(minutes=min_charging_minutes)
    bad = charging[charging['time_taken'] < threshold]
    if not bad.empty:
        report_warning("Some charging blocks are shorter than minimum allowed")
    return bad

def fulfills_timetable(df, timetable_df):
    df_starts = set(df['start time'])
    timetable_starts = set(timetable_df['departure_time'])
    mismatch = timetable_starts - df_starts
    if mismatch:
        report_error(f"Missing timetable start times: {len(mismatch)} unmatched")
        return False, mismatch
    report_info("✅ Timetable matches covered")
    return True, set()