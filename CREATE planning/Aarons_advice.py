import pandas as pd
import datetime
from check_inaccuracies import rename_time_object

def find_insert_charging_and_idle_records(planning_df, distance_matrix, min_idle_duration=15, charging_speed_assumed=450):
    """
    Analyze the existing bus planning to find gaps suitable for charging, simulate charging actions,
    insert them into the planning, and also add idle records for non-movement periods (except at garage).
    :param planning_df: DataFrame containing existing bus schedule data
    :param distance_matrix: DataFrame with distance matrix data
    :param min_idle_duration: Minimum idle time in minutes to consider for charging
    :param charging_speed_assumed: Charging speed in kWh/min
    :return: Updated DataFrame with inserted charging and idle activities
    """
    all_activities = []

    for bus_id in planning_df['bus'].unique():
        bus_trips = planning_df[planning_df['bus'] == bus_id].sort_values(by='end time')

        for i in range(len(bus_trips) - 1):
            current_trip = bus_trips.iloc[i]
            next_trip = bus_trips.iloc[i + 1]

            idle_start = current_trip['end time']
            idle_end = next_trip['start time']
            idle_duration = (idle_end - idle_start).total_seconds() / 60

            # Check and add idle record if idle time exceeds the threshold and not at garage
            if idle_duration > 0 and current_trip['end location'] != 'ehvgar':
                idle_record = {
                    'start location': current_trip['end location'],
                    'end location': current_trip['end location'],
                    'start time': idle_start,
                    'end time': idle_end,
                    'activity': 'idle',
                    'line': 999,
                    'energy consumption': 0,
                    'bus': bus_id
                }
                all_activities.append(idle_record)

            # Check for charging opportunity
            if idle_duration >= min_idle_duration:
                travel_to_garage, _ = lookup_distance_matrix(current_trip['end location'], 'ehvgar', 999, distance_matrix)
                travel_back, _ = lookup_distance_matrix('ehvgar', next_trip['start location'], 999, distance_matrix)
                total_travel_time = travel_to_garage + travel_back

                if idle_duration >= total_travel_time:
                    charging_time = idle_duration - total_travel_time
                    charging_amount = charging_time * charging_speed_assumed

                    # Create records for simulation
                    to_garage_record = {
                        'start location': current_trip['end location'],
                        'end location': 'ehvgar',
                        'start time': idle_start,
                        'end time': idle_start + datetime.timedelta(minutes=travel_to_garage),
                        'activity': 'material trip',
                        'line': 999,
                        'energy consumption': -current_trip['energy consumption'],
                        'bus': bus_id
                    }

                    charging_record = {
                        'start location': 'ehvgar',
                        'end location': 'ehvgar',
                        'start time': idle_start + datetime.timedelta(minutes=travel_to_garage),
                        'end time': idle_end - datetime.timedelta(minutes=travel_back),
                        'activity': 'charging',
                        'line': 999,
                        'energy consumption': charging_amount,
                        'bus': bus_id
                    }

                    return_record = {
                        'start location': 'ehvgar',
                        'end location': next_trip['start location'],
                        'start time': idle_end - datetime.timedelta(minutes=travel_back),
                        'end time': idle_end,
                        'activity': 'material trip',
                        'line': 999,
                        'energy consumption': -charging_amount,
                        'bus': bus_id
                    }

                    all_activities.extend([to_garage_record, charging_record, return_record])

    # Insert all activities into the planning DataFrame
    for activity in all_activities:
        planning_df = planning_df.sort_index().reset_index(drop=True)
        insert_pos = planning_df[planning_df['end time'] <= activity['start time']].index.max() + 1
        planning_df = pd.concat([
            planning_df.iloc[:insert_pos],
            pd.DataFrame([activity]),
            planning_df.iloc[insert_pos:]
        ], ignore_index=True)

    return planning_df

def lookup_distance_matrix(start, end, line, distance_matrix_df):
    if start == 'ehvgar' or end == 'ehvgar':
        row = distance_matrix_df[(distance_matrix_df['start'] == start) & (distance_matrix_df['end'] == end)]
    else:
        row = distance_matrix_df[(distance_matrix_df['start'] == start) & (distance_matrix_df['end'] == end) & (distance_matrix_df['line'] == line)]

    if row.empty:
        raise ValueError(f"Missing entry for: start={start}, end={end}, line={line}.")

    distance_km = int(row.iloc[0]['distance_m']) / 1000
    max_travel_time = int(row.iloc[0]['max_travel_time'])
    energy_use = distance_km * 1.6
    return max_travel_time, energy_use

def main():
    planning = pd.read_excel('../Excel Files/Bus Planning.xlsx')  # Assuming this contains the existing planning
    distance_matrix = pd.read_excel('../Excel Files/DistanceMatrix.xlsx')
    planning = rename_time_object(planning, 'start time', 'end time')
    updated_planning = find_insert_charging_and_idle_records(planning, distance_matrix)

    print("\nUpdated Planning with Charging and Idle Activities:")
    print(updated_planning)

    updated_planning.to_excel("../Excel Files/UpdatedPlanningWithChargingAndIdle.xlsx", index=False)

if __name__ == "__main__":
    main()