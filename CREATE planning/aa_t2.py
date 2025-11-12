import pandas as pd
import datetime
from tqdm import tqdm
from check_inaccuracies import rename_time_object


def create_distance_lookup(distance_matrix_df):
    """Create a dictionary for O(1) distance matrix lookups."""
    lookup = {}
    for _, row in distance_matrix_df.iterrows():
        key = (row['start'], row['end'], row['line'])
        lookup[key] = {
            'distance_m': int(row['distance_m']),
            'max_travel_time': int(row['max_travel_time'])
        }
    return lookup


def lookup_distance_matrix(start, end, line, distance_lookup):
    """Fast lookup using pre-built dictionary."""
    key = (start, end, line)
    if key not in distance_lookup:
        raise ValueError(f"Missing entry for: start={start}, end={end}, line={line}.")
    data = distance_lookup[key]
    distance_km = data['distance_m'] / 1000
    max_travel_time = data['max_travel_time']
    energy_use = distance_km * 1.6
    return max_travel_time, energy_use


def find_insert_charging_and_idle_records(planning_df, distance_matrix, min_idle_duration=15, charging_speed_assumed=450):
    """
    Finds idle periods and inserts charging records if the idle time is sufficient.
    A bus will only be sent to charge if the resulting charging time is at least `min_idle_duration`.
    """
    distance_lookup = create_distance_lookup(distance_matrix)
    all_activities = []
    bus_ids = planning_df['bus'].unique()

    for bus_id in tqdm(bus_ids, desc="Processing buses for charging/idle"):
        bus_trips = planning_df[planning_df['bus'] == bus_id].sort_values(by='end time').reset_index(drop=True)

        for i in range(len(bus_trips) - 1):
            current_trip = bus_trips.iloc[i]
            next_trip = bus_trips.iloc[i + 1]

            idle_start = current_trip['end time']
            idle_end = next_trip['start time']
            idle_duration = (idle_end - idle_start).total_seconds() / 60

            # First, create an 'idle' record if the bus is not at the garage and there is any idle time.
            # This will be overwritten if a charging session is inserted.
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

            # Now, check if a charging session is feasible and worthwhile
            if idle_duration >= min_idle_duration:
                travel_to_garage, _ = lookup_distance_matrix(current_trip['end location'], 'ehvgar', 999, distance_lookup)
                travel_back, _ = lookup_distance_matrix('ehvgar', next_trip['start location'], 999, distance_lookup)
                total_travel_time = travel_to_garage + travel_back

                if idle_duration >= total_travel_time:
                    charging_time = idle_duration - total_travel_time

                    # --- MODIFICATION ---
                    # Only proceed with charging if the calculated charging time is at least the minimum duration.
                    # This prevents short, inefficient charging trips.
                    if charging_time >= min_idle_duration:
                        charging_amount = charging_time * charging_speed_assumed

                        # Before adding the new records, remove the idle record that was added earlier for this period
                        if all_activities and all_activities[-1]['bus'] == bus_id and \
                           all_activities[-1]['activity'] == 'idle' and \
                           all_activities[-1]['start time'] == idle_start and \
                           all_activities[-1]['end time'] == idle_end:
                            all_activities.pop()

                        to_garage_record = {
                            'start location': current_trip['end location'],
                            'end location': 'ehvgar',
                            'start time': idle_start,
                            'end time': idle_start + datetime.timedelta(minutes=travel_to_garage),
                            'activity': 'material trip',
                            'line': 999,
                            'energy consumption': -current_trip['energy consumption'], # Assuming energy use is proportional to distance
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
                            'energy consumption': -charging_amount, # Assuming energy use is proportional to distance
                            'bus': bus_id
                        }

                        all_activities.extend([to_garage_record, charging_record, return_record])

    if all_activities:
        activities_df = pd.DataFrame(all_activities)
        planning_df = pd.concat([planning_df, activities_df], ignore_index=True)
        planning_df = planning_df.sort_values(by=['bus', 'start time']).reset_index(drop=True)

    return planning_df


def heuristic_bus_assignment(planning_df, distance_matrix):
    """[Unchanged from your provided code]"""
    print("Starting heuristic bus assignment...")
    distance_lookup = create_distance_lookup(distance_matrix)
    trips = planning_df.to_dict(orient='records')
    trips_sorted = sorted(trips, key=lambda x: x['start time'])
    buses = []
    bus_assignments = []

    for trip in tqdm(trips_sorted, desc="Assigning trips to buses"):
        compatible_buses = []

        for bus in buses:
            last_trip = bus['last_trip']
            if last_trip['end time'] <= trip['start time']:
                try:
                    travel_time = calculate_travel_time(
                        last_trip['end location'], trip['start location'], distance_lookup
                    )
                    arrival_time = last_trip['end time'] + datetime.timedelta(minutes=travel_time)
                    if arrival_time <= trip['start time']:
                        compatible_buses.append({
                            'bus': bus,
                            'arrival_time': arrival_time,
                            'idle_time': (trip['start time'] - arrival_time).total_seconds() / 60
                        })
                except ValueError:
                    continue

        if compatible_buses:
            best_match = min(compatible_buses, key=lambda x: x['idle_time'])
            chosen_bus = best_match['bus']
            bus_id = chosen_bus['id']
            chosen_bus['last_trip'] = trip
        else:
            bus_id = len(buses) + 1
            new_bus = {'id': bus_id, 'last_trip': trip}
            buses.append(new_bus)

        bus_assignments.append((trip, bus_id))

    improved_planning = []
    for trip, bus_id in bus_assignments:
        trip_copy = trip.copy()
        trip_copy['bus'] = bus_id
        improved_planning.append(trip_copy)

    result_df = pd.DataFrame(improved_planning)
    result_df = result_df.sort_values(by=['bus', 'start time']).reset_index(drop=True)
    print(f"Number of buses used: {len(buses)}")
    print(f"Original number of buses: {planning_df['bus'].nunique()}")
    print(f"Reduction: {planning_df['bus'].nunique() - len(buses)} buses")
    return result_df


def calculate_travel_time(start_loc, end_loc, distance_lookup):
    """Calculate travel time using pre-built lookup."""
    travel_time, _ = lookup_distance_matrix(start_loc, end_loc, 999, distance_lookup)
    return travel_time


def ensure_garage_start_end(planning_df, distance_lookup):
    """
    Ensure each bus's schedule starts and ends at the garage ('ehvgar').
    Adds material trips to/from garage if necessary, even across midnight.
    """
    if planning_df.empty:
        return planning_df

    bus_ids = planning_df['bus'].unique()
    all_activities = []

    for bus_id in tqdm(bus_ids, desc="Ensuring garage start/end for buses"):
        bus_trips = planning_df[planning_df['bus'] == bus_id].sort_values(by='start time').reset_index(drop=True)
        if bus_trips.empty:
            continue

        # Add trip FROM garage to first activity if needed
        first_trip = bus_trips.iloc[0]
        if first_trip['start location'] != 'ehvgar':
            start_loc = 'ehvgar'
            end_loc = first_trip['start location']
            try:
                travel_time, energy_use = lookup_distance_matrix(start_loc, end_loc, 999, distance_lookup)
            except ValueError as e:
                print(f"Missing distance for {start_loc} -> {end_loc}: {e}")
                travel_time = 0
                energy_use = 0

            # Calculate start time for the new trip (must arrive exactly at first_trip's start time)
            new_start_time = first_trip['start time'] - datetime.timedelta(minutes=travel_time)
            new_end_time = first_trip['start time']

            to_garage_record = {
                'start location': start_loc,
                'end location': end_loc,
                'start time': new_start_time,
                'end time': new_end_time,
                'activity': 'material trip',
                'line': 999,
                'energy consumption': energy_use,
                'bus': bus_id
            }
            all_activities.append(to_garage_record)

        # Add trip TO garage from last activity if needed
        last_trip = bus_trips.iloc[-1]
        if last_trip['end location'] != 'ehvgar':
            start_loc = last_trip['end location']
            end_loc = 'ehvgar'
            try:
                travel_time, energy_use = lookup_distance_matrix(start_loc, end_loc, 999, distance_lookup)
            except ValueError as e:
                print(f"Missing distance for {start_loc} -> {end_loc}: {e}")
                travel_time = 0
                energy_use = 0

            # Calculate start/end times for the new trip (departs immediately after last activity)
            new_start_time = last_trip['end time']
            new_end_time = last_trip['end time'] + datetime.timedelta(minutes=travel_time)

            return_garage_record = {
                'start location': start_loc,
                'end location': end_loc,
                'start time': new_start_time,
                'end time': new_end_time,
                'activity': 'material trip',
                'line': 999,
                'energy consumption': energy_use,
                'bus': bus_id
            }
            all_activities.append(return_garage_record)

    # Add new activities to the DataFrame and re-sort
    if all_activities:
        new_activities_df = pd.DataFrame(all_activities)
        planning_df = pd.concat([planning_df, new_activities_df], ignore_index=True)
        planning_df = planning_df.sort_values(by=['bus', 'start time']).reset_index(drop=True)

    return planning_df


def main():
    print("Loading data...")
    planning = pd.read_excel('../Excel Files/Bus Planning.xlsx')
    distance_matrix = pd.read_excel('../Excel Files/DistanceMatrix.xlsx')
    planning = rename_time_object(planning, 'start time', 'end time')

    print(f"\nOriginal planning has {len(planning)} trips and {planning['bus'].nunique()} buses")

    # Step 1: Heuristic bus assignment
    print("\nImproving bus assignments using heuristic...")
    distance_lookup = create_distance_lookup(distance_matrix)
    improved_planning = heuristic_bus_assignment(planning, distance_matrix)

    # Step 2: Insert charging/idle records
    print("\nInserting charging and idle records...")
    improved_with_charging = find_insert_charging_and_idle_records(improved_planning, distance_matrix)

    # Step 3: Ensure every bus starts/ends at garage (even past midnight)
    print("\nEnsuring buses start and end at garage...")
    final_planning = ensure_garage_start_end(improved_with_charging, distance_lookup)

    print(f"\nFinal planning has {len(final_planning)} records")
    print("\nSample of final planning:")
    print(final_planning.head(20))

    print("\nSaving results...")
    final_planning.to_excel("../Excel Files/IImprovedBusPlanning.xlsx", index=False)

    summary = {
        'Total Trips (Original)': [len(planning)],
        'Original Buses': [planning['bus'].nunique()],
        'Final Buses': [final_planning['bus'].nunique()],
        'Final Records (with charging/idle/garage)': [len(final_planning)]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel("../Excel Files/ImprovementSummary.xlsx", index=False)

    print("\nDone!")


import pandas as pd
import datetime
from tqdm import tqdm
from check_inaccuracies import rename_time_object
from typing import Tuple, List


# [Previous helper functions remain the same until find_insert_charging_and_idle_records]

def calculate_energy_consumption(activity: str, start_loc: str, end_loc: str,
                                 distance_lookup: dict, charging_speed: float,
                                 time_taken: datetime.timedelta) -> float:
    """Calculate energy consumption for a given activity."""
    minutes = time_taken.total_seconds() / 60

    if activity == 'charging':
        return charging_speed * minutes  # Positive for charging
    elif activity in ('material trip', 'service trip'):
        try:
            distance_km = lookup_distance_matrix(start_loc, end_loc, 999, distance_lookup)[0] / 1000
            return distance_km * 1.6  # Negative for consumption
        except ValueError:
            print(f"Warning: Missing distance data for {start_loc} -> {end_loc}")
            return 0
    elif activity == 'idle':
        return 0.5 * (minutes / 60)  # Small idle consumption
    return 0


def track_energy_levels(planning_df: pd.DataFrame, distance_lookup: dict,
                        initial_energy: float = 255, min_energy: float = 30) -> pd.DataFrame:
    """
    Track energy levels throughout each bus's schedule and add charging if needed.
    Returns the updated schedule with energy levels maintained above minimum.
    """
    planning_df = planning_df.copy()
    planning_df['energy_after'] = 0.0
    buses = planning_df['bus'].unique()
    updated_activities = []

    for bus_id in tqdm(buses, desc="Tracking energy levels"):
        bus_trips = planning_df[planning_df['bus'] == bus_id].sort_values('start time').reset_index(drop=True)
        current_energy = initial_energy
        new_activities = []

        for idx, row in bus_trips.iterrows():
            # Calculate energy consumption for this activity
            time_taken = row['end time'] - row['start time']
            energy_consumed = calculate_energy_consumption(
                row['activity'], row['start location'], row['end location'],
                distance_lookup, 450, time_taken
            )

            # Check if we need to charge before this activity
            if current_energy - energy_consumed < min_energy and row['activity'] != 'charging':
                # Try to find charging opportunities in previous idle time
                for prev_idx, prev_row in reversed(list(enumerate(new_activities))):
                    if (prev_row['activity'] == 'idle' and
                            prev_row['start location'] == 'ehvgar' and
                            (prev_row['end time'] - prev_row['start time']).total_seconds() >= 15 * 60):
                        # Convert idle to charging
                        charge_time = (prev_row['end time'] - prev_row['start time']).total_seconds() / 60
                        charge_energy = 450 * (charge_time / 60)

                        new_activities[prev_idx] = {
                            **prev_row,
                            'activity': 'charging',
                            'energy consumption': charge_energy
                        }
                        current_energy += charge_energy
                        break

                # If still not enough energy, force a charging session
                if current_energy - energy_consumed < min_energy:
                    charge_needed = min_energy - (current_energy - energy_consumed) + 10  # Small buffer
                    charge_time = (charge_needed / 450) * 60  # Minutes

                    # Add travel to garage if not already there
                    if row['start location'] != 'ehvgar':
                        travel_time, _ = lookup_distance_matrix(row['start location'], 'ehvgar', 999, distance_lookup)
                        to_garage = {
                            'start location': row['start location'],
                            'end location': 'ehvgar',
                            'start time': row['start time'] - datetime.timedelta(minutes=travel_time + charge_time),
                            'end time': row['start time'] - datetime.timedelta(minutes=charge_time),
                            'activity': 'material trip',
                            'line': 999,
                            'bus': bus_id,
                            'energy consumption': 0  # Will be calculated later
                        }
                        new_activities.append(to_garage)

                    # Add charging session
                    charge_start = row['start time'] - datetime.timedelta(minutes=charge_time)
                    charging = {
                        'start location': 'ehvgar',
                        'end location': 'ehvgar',
                        'start time': charge_start,
                        'end time': row['start time'],
                        'activity': 'charging',
                        'line': 999,
                        'bus': bus_id,
                        'energy consumption': charge_needed
                    }
                    new_activities.append(charging)
                    current_energy += charge_needed

            # Update energy level
            current_energy -= energy_consumed
            new_row = row.to_dict()
            new_row['energy_after'] = current_energy
            new_activities.append(new_row)

        updated_activities.extend(new_activities)

    return pd.DataFrame(updated_activities)


def check_energy_consumption(
        df: pd.DataFrame,
        distancematrix: pd.DataFrame,
        idle_cost_ph: float,
        charge_speed_assumed: float,
        low_charge_rate: float,
        high_charge_rate: float,
        low_energy_use: float,
        high_energy_use: float,
        low_idle_cost: float,
        high_idle_cost: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate per-row energy consumption against expected physical ranges, summarizing all row indices per error type."""
    errors = []
    df = df.copy()
    distance_lookup = distancematrix.set_index(['start', 'end'])['distance_m'].to_dict()

    # Collect row indices by error category
    invalid_charging, invalid_trip, invalid_idle = [], [], []
    unknown_trip, unknown_activity = [], []
    generic_errors = []

    for idx, row in df.iterrows():
        try:
            activity = row['activity']
            energy = row['energy consumption']
            minutes = row['time_taken'].total_seconds() / 60

            if activity == 'charging':
                low = -charge_speed_assumed * minutes * low_charge_rate
                high = -charge_speed_assumed * minutes * high_charge_rate
                if not (high <= energy <= low):
                    invalid_charging.append(idx)
                    df.at[idx, 'energy consumption'] = -charge_speed_assumed * minutes

            elif activity in ('material trip', 'service trip'):
                km = distance_lookup.get((row['start location'], row['end location']))
                if km is None:
                    unknown_trip.append(idx)
                    continue
                km /= 1000
                low = low_energy_use * km
                high = high_energy_use * km
                if not (low <= energy <= high):
                    invalid_trip.append(idx)
                    df.at[idx, 'energy consumption'] = km * ((low_energy_use + high_energy_use) / 2)

            elif activity == 'idle':
                base = (idle_cost_ph / 60) * minutes
                low = base * low_idle_cost
                high = base * high_idle_cost
                if not (low <= energy <= high):
                    invalid_idle.append(idx)
                    df.at[idx, 'energy consumption'] = base

            else:
                unknown_activity.append(idx)

        except Exception as e:
            generic_errors.append((idx, str(e)))

    # Summarize all errors in compact messages
    if invalid_charging:
        errors.append(f"Charging energy outside expected range at rows {invalid_charging}, this is fixed automatically")
    if invalid_trip:
        errors.append(f"Trip energy outside expected range at rows {invalid_trip}, this is fixed automatically")
    if invalid_idle:
        errors.append(f"Idle energy outside expected range at rows {invalid_idle}, this is fixed automatically")
    if unknown_trip:
        errors.append(f"Unknown trip start/end combination at rows {unknown_trip}")
    if unknown_activity:
        errors.append(f"Unknown activity type at rows {unknown_activity}")
    if generic_errors:
        formatted = [f"Row {i}: {msg}" for i, msg in generic_errors]
        errors.append("Unexpected validation errors:\n" + "\n".join(formatted))

    return df, errors



def main():
    print("Loading data...")
    planning = pd.read_excel('../Excel Files/Bus Planning.xlsx')
    distance_matrix = pd.read_excel('../Excel Files/DistanceMatrix.xlsx')
    planning = rename_time_object(planning, 'start time', 'end time')

    print(f"\nOriginal planning has {len(planning)} trips and {planning['bus'].nunique()} buses")

    # Step 1: Heuristic bus assignment
    print("\nImproving bus assignments using heuristic...")
    distance_lookup = create_distance_lookup(distance_matrix)
    improved_planning = heuristic_bus_assignment(planning, distance_matrix)

    # Step 2: Insert charging/idle records
    print("\nInserting charging and idle records...")
    improved_with_charging = find_insert_charging_and_idle_records(improved_planning, distance_matrix)

    # Step 3: Ensure energy levels are maintained
    print("\nTracking and maintaining energy levels...")
    final_planning = track_energy_levels(improved_with_charging, distance_lookup)

    # Step 4: Validate energy consumption
    print("\nValidating energy consumption...")
    final_planning, errors = check_energy_consumption(
        final_planning,
        distance_matrix,
        idle_cost_ph=5.0,
        charge_speed_assumed=450
    )

    # Step 5: Ensure garage start/end
    print("\nEnsuring buses start and end at garage...")
    final_planning = ensure_garage_start_end(final_planning, distance_lookup)

    # Print any validation errors
    if errors:
        print("\nEnergy validation warnings:")
        for error in errors:
            print(f"  - {error}")

    print(f"\nFinal planning has {len(final_planning)} records")
    print("\nSample of final planning:")
    print(final_planning.head(20))

    print("\nSaving results...")
    final_planning.to_excel("../Excel Files/IImprovedBusPlanning.xlsx", index=False)

    # Add summary statistics
    summary = {
        'Total Trips (Original)': [len(planning)],
        'Original Buses': [planning['bus'].nunique()],
        'Final Buses': [final_planning['bus'].nunique()],
        'Final Records': [len(final_planning)],
        'Min Energy Level': [final_planning.groupby('bus')['energy_after'].min().min()],
        'Max Energy Level': [final_planning.groupby('bus')['energy_after'].max().max()],
        'Energy Validation Errors': [len(errors)]
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_excel("../Excel Files/ImprovementSummary.xlsx", index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()