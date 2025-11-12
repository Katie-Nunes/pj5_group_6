"""
Bus Planning Feasibility Checks

This module provides functions to validate bus planning schedules against
operational constraints including battery SOC limits, depot requirements,
and timetable coverage.
"""
import pandas as pd
import streamlit as st
from typing import Tuple, Set

from logging_utils import report_error, report_warning, report_info
from check_inaccuracies import rename_time_object


def energy_state(
    df: pd.DataFrame,
    full_new_battery: float = 300,
    state_of_health_frac: float = 0.85,
    starting_soc_percentage: float = 0.9
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate current battery charge for each row in the bus planning schedule.
    
    Args:
        df: Bus planning dataframe with 'bus' and 'energy consumption' columns
        full_new_battery: Full battery capacity when new (kWh)
        state_of_health_frac: Battery degradation factor (0-1)
        starting_soc_percentage: Starting SOC as percentage of effective capacity
    
    Returns:
        Tuple of (updated dataframe with 'current_charge' column, effective_capacity)
    """
    try:
        effective_capacity = full_new_battery * state_of_health_frac  # 255 kWh
        initial_charge = effective_capacity * starting_soc_percentage  # 229.5 kWh (buses start at 90%)
        df = df.copy()
        df['cumulative_energy_used'] = df.groupby('bus')['energy consumption'].cumsum()
        df['current_charge'] = initial_charge - df['cumulative_energy_used']
        return df, effective_capacity  # Return effective_capacity for threshold calculation
    except Exception as e:
        report_error("Error computing energy state", e)
        return df, 0


def check_energy_feasibility(
    df: pd.DataFrame,
    initial_charge: float,
    low: float = 0.1,
    high: float = 0.9
) -> bool:
    """
    Validate that all buses stay within battery SOC constraints.
    
    Args:
        df: Dataframe with 'current_charge' column
        initial_charge: Effective battery capacity (kWh)
        low: Minimum SOC threshold as fraction of capacity
        high: Maximum SOC threshold as fraction of capacity
    
    Returns:
        True if all buses stay within SOC limits, False otherwise
    """
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


def validate_start_end_locations(
    df: pd.DataFrame,
    start_end_location: str = "ehvgar"
) -> pd.DataFrame:
    """
    Check that all buses start and end their schedule at the depot.
    
    Args:
        df: Bus planning dataframe with 'bus', 'start location', 'end location'
        start_end_location: Expected depot location code
    
    Returns:
        Dataframe of buses that don't start/end at depot (empty if all valid)
    """
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
        report_warning(f"Some buses do not start/end at depot. Failed at rows: {failed_indices}")
    return not_ok


def minimum_charging(
    df: pd.DataFrame,
    min_charging_minutes: int = 15
) -> pd.DataFrame:
    """
    Check that all charging sessions meet minimum duration requirement.
    
    Args:
        df: Bus planning dataframe with 'activity' and 'time_taken' columns
        min_charging_minutes: Minimum required charging duration (minutes)
    
    Returns:
        Dataframe of charging sessions below minimum (empty if all valid)
    """
    charging = df[df['activity'] == 'charging']
    threshold = pd.Timedelta(minutes=min_charging_minutes)
    bad = charging[charging['time_taken'] < threshold]
    if not bad.empty:
        failed_indices = bad.index.tolist()
        report_warning(f"Some charging blocks are shorter than minimum allowed. Failed at rows: {failed_indices}")
    return bad


def fulfills_timetable(
    df: pd.DataFrame,
    timetable_df: pd.DataFrame
) -> Tuple[bool, Set]:
    """
    Verify that all timetabled trips are covered in the bus planning schedule.
    
    Args:
        df: Bus planning dataframe with service trips
        timetable_df: Master timetable with all required trips
    
    Returns:
        Tuple of (all_covered: bool, missing_trips: set)
    """
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
        report_error(f"Missing {len(missing_trips)} timetable trips from service schedule")
        return False, missing_trips

    report_info("ᕙ(  •̀ ᗜ •́  )ᕗ Timetable matches covered")
    return True, set()


def check_all_feasibility(
    df: pd.DataFrame,
    timetable_df: pd.DataFrame
) -> Tuple[bool, Set]:
    """
    Run all feasibility checks on a bus planning schedule.
    
    Validates:
    - Battery SOC stays within 10-90% limits
    - All buses start and end at depot
    - Charging sessions meet minimum duration
    - All timetabled trips are covered
    
    Args:
        df: Bus planning schedule dataframe
        timetable_df: Master timetable with required trips
    
    Returns:
        Tuple of (is_feasible: bool, missing_trips: set)
    """
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