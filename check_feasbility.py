import pandas as pd
from check_inaccuracies import rename_time_object

def energy_state(df, full_new_battery, state_of_health_frac):
    initial_charge = full_new_battery * state_of_health_frac

    df['cumulative_energy_used'] = df.groupby('bus')['energy consumption'].cumsum()
    df['current_charge'] = initial_charge - df['cumulative_energy_used']

    df.drop(columns=['cumulative_energy_consumption'], inplace=True, errors='ignore')
    return df, initial_charge

def check_energy_feasibility(df, initial_charge, low, high):
    min_bat_life = initial_charge * low
    max_charging = initial_charge * high

    charging_rows = df.loc[df['energy consumption'] < 0]
    under_min = df.loc[df['current_charge'] < min_bat_life]
    over_max = charging_rows.loc[charging_rows['energy consumption'] > -max_charging]  # Adjusted condition

    if not under_min.empty:
        print(f"Error: Some trips fall below minimum battery threshold ({min_bat_life:.2f} kWh):")
        print(f"{under_min[['bus', 'start time', 'current_charge']]}")
        return False

    print("All trips are charge feasible.")
    if not over_max.empty:
        print(f"Error: Some buses exceed max charging limit ({max_charging:.2f} kWh):")
        print(over_max[['bus', 'start time', 'cumulative_energy_gained']])
        return False

    print("All trips are energy-feasible.")
    return True

def validate_start_end_locations(df, start_end_location='ehvgar'):
    start_locations = df.groupby('bus', as_index=False).first()['start location']
    end_locations = df.groupby('bus', as_index=False).last()['end location']

    bus_start_end = pd.DataFrame({'bus': df.groupby('bus', as_index=False).first()['bus'], 'start_location': start_locations, 'end_location': end_locations})

    invalid_start = bus_start_end[bus_start_end['start_location'] != start_end_location]
    invalid_end = bus_start_end[bus_start_end['end_location'] != start_end_location]

    invalid_buses = pd.merge(invalid_start, invalid_end, on='bus', how='outer', suffixes=('_start', '_end'))

    if not invalid_buses.empty:
        print("Error: The following buses do not start/end at 'ehvgar':")
        print(invalid_buses)
    else:
        print("All buses start and end at 'ehvgar'.")
    return invalid_buses


def minimum_charging(df, min_ch_time_seconds):
    charging_rows = df[df['activity'] == 'charging']
    threshold = pd.Timedelta(seconds=min_ch_time_seconds)
    insufficient_charging = charging_rows[charging_rows['time_taken'] < threshold]

    if not insufficient_charging.empty:
        return "Ya got charging less than allowed time, change minimum charging time in file or value checked against here"
    return "All gud on min chargin bruv"

def fulfills_timetable(df, timetable_df):
    df_starts = set(df['start time'])
    timetable_starts = set(timetable_df['departure_time'])
    mismatched_starts = timetable_starts - df_starts
    is_valid = len(mismatched_starts) == 0
    return is_valid, mismatched_starts

def check_feasibility(df, full_new_battery, state_of_health_frac, timetable_df):
    df, initial_charge = energy_state(df, full_new_battery, state_of_health_frac)
    check_energy_feasibility(df, initial_charge, 0.1, 0.9)
    invalid_buses = validate_start_end_locations(df)
    print(invalid_buses)
    print(minimum_charging(df, 900))
    rename_time_object(timetable_df, 'departure_time', None)
    is_valid, mismatched_starts = fulfills_timetable(df, timetable_df)
    return df


















