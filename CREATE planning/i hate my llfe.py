import pandas as pd
import datetime
import random
import math
from check_inaccuracies import rename_time_object
from copy import deepcopy
from tqdm import tqdm


def swap_bus_trips(planning_df, trip1_idx, trip2_idx):
    """
    Swap two trips between different buses if feasible.
    Returns a new DataFrame with the swap if feasible, otherwise returns None.
    """
    # Create a copy to avoid modifying the original
    df = planning_df.copy()

    # Get the two trips to swap
    trip1 = df.iloc[trip1_idx].copy()
    trip2 = df.iloc[trip2_idx].copy()

    # Can't swap trips of the same bus
    if trip1['bus'] == trip2['bus']:
        return None

    # Store original bus assignments
    original_bus1 = trip1['bus']
    original_bus2 = trip2['bus']

    # Temporarily assign to different buses to check feasibility
    df.at[trip1_idx, 'bus'] = original_bus2
    df.at[trip2_idx, 'bus'] = original_bus1

    # Sort by bus and start time to check for conflicts
    df_sorted = df.sort_values(['bus', 'start time']).reset_index(drop=True)

    # Check for conflicts in both buses' schedules
    for bus_id in [original_bus1, original_bus2]:
        bus_trips = df_sorted[df_sorted['bus'] == bus_id].sort_values('start time')

        # Check for time overlaps
        for i in range(len(bus_trips) - 1):
            current = bus_trips.iloc[i]
            next_trip = bus_trips.iloc[i + 1]

            # Check if end time of current trip is after start time of next trip
            if current['end time'] > next_trip['start time']:
                return None  # Conflict found

    # If we get here, the swap is feasible
    return df


def calculate_schedule_cost(planning_df, distance_matrix):
    """
    Calculate a cost for the schedule (lower is better).
    This is a simple example that considers total energy consumption and idle time.
    """
    total_energy = planning_df['energy consumption'].sum()
    total_idle_time = 0
    total_trips = len(planning_df)

    # Calculate total idle time
    for bus_id in planning_df['bus'].unique():
        bus_trips = planning_df[planning_df['bus'] == bus_id].sort_values('start time')
        for i in range(len(bus_trips) - 1):
            idle_time = (bus_trips.iloc[i + 1]['start time'] - bus_trips.iloc[i]['end time']).total_seconds() / 60
            total_idle_time += max(0, idle_time)

    # Penalize negative energy consumption (charging) less than positive (consumption)
    energy_penalty = sum([e if e > 0 else e * 0.5 for e in planning_df['energy consumption']])

    # Simple weighted sum (adjust weights as needed)
    cost = energy_penalty + (total_idle_time * 0.1)
    return cost


def simulated_annealing(planning_df, distance_matrix, initial_temp=1000, cooling_rate=0.995, min_temp=0.1,
                        iterations_per_temp=100):
    """
    Simulated annealing implementation for bus trip swapping.

    Parameters:
    - initial_temp: Starting temperature
    - cooling_rate: Rate at which temperature decreases (0.995 means 0.5% reduction per temperature step)
    - min_temp: Minimum temperature at which to stop
    - iterations_per_temp: Number of iterations at each temperature level
    """
    current_df = planning_df.copy()
    current_cost = calculate_schedule_cost(current_df, distance_matrix)
    best_df = current_df.copy()
    best_cost = current_cost

    temp = initial_temp
    iteration = 0

    print(f"Initial cost: {best_cost}")

    while temp > min_temp:
        improvements = 0
        # Add tqdm progress bar for iterations at current temperature
        for _ in tqdm(range(iterations_per_temp), desc=f"Temp {temp:.2f}", leave=False):
            # Get two random trips to swap
            trip1_idx = random.randint(0, len(current_df) - 1)
            trip2_idx = random.randint(0, len(current_df) - 1)

            # Skip if same trip
            if trip1_idx == trip2_idx:
                continue

            # Try the swap
            new_df = swap_bus_trips(current_df, trip1_idx, trip2_idx)

            if new_df is not None:
                # Calculate new cost
                new_cost = calculate_schedule_cost(new_df, distance_matrix)

                # Calculate acceptance probability
                if new_cost < current_cost:
                    # Always accept better solutions
                    current_df = new_df
                    current_cost = new_cost
                    improvements += 1

                    # Update best if better
                    if new_cost < best_cost:
                        best_df = new_df.copy()
                        best_cost = new_cost
                        print(f"Iteration {iteration}: New best cost: {best_cost} (Temp: {temp:.2f})")
                else:
                    # Sometimes accept worse solutions based on temperature and how much worse
                    cost_difference = new_cost - current_cost
                    acceptance_probability = math.exp(-cost_difference / temp)

                    if random.random() < acceptance_probability:
                        current_df = new_df
                        current_cost = new_cost

            iteration += 1

        # Cool down
        temp *= cooling_rate

        # Print progress
        if iteration % 10 == 0:
            print(
                f"Iteration {iteration}: Temp={temp:.2f}, Current Cost={current_cost}, Best Cost={best_cost}, Improvements={improvements}/{iterations_per_temp}")

    print(f"\nOptimization complete after {iteration} iterations")
    print(f"Final best cost: {best_cost}")
    return best_df


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

import pandas as pd
import datetime

def ensure_activity_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'activity' not in df.columns:
        df = df.copy()
        df['activity'] = 'service'  # mark all existing rows as service by default
    else:
        # Normalize empty/NaN as service
        df['activity'] = df['activity'].fillna('service')
    return df

def add_material_trip(df: pd.DataFrame, bus_id, start_loc, end_loc, start_time, end_time, energy_use) -> pd.DataFrame:
    """Append a material (deadhead) trip row."""
    row = {
        'start location': start_loc,
        'end location': end_loc,
        'start time': start_time,
        'end time': end_time,
        'activity': 'material trip',
        'line': 999,
        'energy consumption': energy_use,  # positive consumption for driving
        'bus': bus_id
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def compute_service_minutes(df: pd.DataFrame) -> dict:
    svc = df[df['activity'] == 'service'].copy()
    if svc.empty:
        return {}
    svc['duration_min'] = (svc['end time'] - svc['start time']).dt.total_seconds() / 60
    return svc.groupby('bus')['duration_min'].sum().to_dict()

def last_event_before(df_bus: pd.DataFrame, t: pd.Timestamp):
    prev = df_bus[df_bus['end time'] <= t].sort_values('end time').tail(1)
    if prev.empty:
        return None
    return prev.iloc[0]

def first_event_after(df_bus: pd.DataFrame, t: pd.Timestamp):
    nxt = df_bus[df_bus['start time'] >= t].sort_values('start time').head(1)
    if nxt.empty:
        return None
    return nxt.iloc[0]

def can_cover_with_repositions(df: pd.DataFrame,
                               bus_id,
                               trip_row,
                               distance_matrix_df,
                               buffer_minutes=2):
    """
    Check if bus_id can cover trip_row by adding a reposition before (from its last end)
    and a reposition after (to its next start), without breaking existing schedule.
    Returns None if not possible, or a dict with timings/energy for the two repositions.
    """
    df_bus = df[df['bus'] == bus_id].sort_values('start time')

    # Prior and next events around the candidate trip
    prev_ev = last_event_before(df_bus, trip_row['start time'])
    next_ev = first_event_after(df_bus, trip_row['end time'])

    if prev_ev is None:
        prev_loc = 'ehvgar'  # assume buses with no prior tasks are at garage
        prev_end_time = pd.Timestamp.min  # available since the day start
    else:
        prev_loc = prev_ev['end location']
        prev_end_time = prev_ev['end time']

    # Reposition BEFORE: prev_loc -> trip start
    try:
        t_before_min, e_before = lookup_distance_matrix(prev_loc, trip_row['start location'], 999, distance_matrix_df)
    except Exception:
        return None
    depart_time = trip_row['start time'] - datetime.timedelta(minutes=t_before_min)

    # Ensure no overlap with existing tasks (including material trips we might have added)
    # The bus must be free in [depart_time, trip_row['end time'])
    df_bus_overlap = df_bus[(df_bus['start time'] < trip_row['end time']) &
                            (df_bus['end time'] > depart_time)]
    if not df_bus_overlap.empty:
        return None
    if prev_end_time > depart_time - datetime.timedelta(minutes=buffer_minutes):
        return None

    # Reposition AFTER: from trip end -> next_ev start (if any)
    t_after_min = 0
    e_after = 0
    next_start_loc = None
    if next_ev is not None:
        next_start_loc = next_ev['start location']
        try:
            t_after_min, e_after = lookup_distance_matrix(trip_row['end location'], next_start_loc, 999, distance_matrix_df)
        except Exception:
            return None
        arrive_next = trip_row['end time'] + datetime.timedelta(minutes=t_after_min)
        if arrive_next > next_ev['start time'] - datetime.timedelta(minutes=buffer_minutes):
            return None

        # Also ensure no overlap during [trip_end, trip_end + t_after)
        df_after_overlap = df_bus[(df_bus['start time'] < arrive_next) &
                                  (df_bus['end time'] > trip_row['end time'])]
        if not df_after_overlap.empty:
            return None

    return {
        'prev_loc': prev_loc,
        't_before_min': t_before_min,
        'e_before': e_before,
        'depart_time': depart_time,
        'next_start_loc': next_start_loc,
        't_after_min': t_after_min,
        'e_after': e_after
    }

def swap_buses_for_charging(planning_df: pd.DataFrame,
                            distance_matrix_df: pd.DataFrame,
                            prefer_garage_only=False,
                            buffer_minutes=2,
                            require_less_used=True) -> pd.DataFrame:
    """
    Greedy pass: for each bus's consecutive pair (current, next), try to reassign 'next'
    to another less-used bus that can reach it and still make its own next work.
    Inserts material trips for the swapped-in bus before and after the reassigned trip.
    """
    df = ensure_activity_column(planning_df).copy()
    df = df.sort_values('start time').reset_index(drop=False)  # preserve original index to edit rows safely
    df.rename(columns={'index': '_orig_idx'}, inplace=True)

    # Static "less used" metric over the day (service minutes). You can switch to energy if preferred.
    service_minutes = compute_service_minutes(df)

    changed = True
    while changed:
        changed = False

        for bus_id in df['bus'].unique():
            # Fresh view of bus timeline
            my = df[df['bus'] == bus_id].sort_values('start time')
            if len(my) < 2:
                continue

            for i in range(len(my) - 1):
                next_trip = my.iloc[i + 1]
                # Only consider swapping service rows
                if next_trip['activity'] != 'service':
                    continue

                # Build candidate pool
                candidates = []
                for other_bus in df['bus'].unique():
                    if other_bus == bus_id:
                        continue

                    # Optionally restrict to buses that are currently at garage before the reassigned trip
                    preview = df[df['bus'] == other_bus].sort_values('start time')
                    prev_ev = last_event_before(preview, next_trip['start time'])
                    if prefer_garage_only:
                        prior_loc = 'ehvgar' if prev_ev is None else prev_ev['end location']
                        if prior_loc != 'ehvgar':
                            continue

                    fit = can_cover_with_repositions(df, other_bus, next_trip, distance_matrix_df, buffer_minutes=buffer_minutes)
                    if fit is None:
                        continue

                    candidates.append((
                        other_bus,
                        service_minutes.get(other_bus, 0.0),  # less used first
                        fit
                    ))

                if not candidates:
                    continue

                # Choose least-used (tie-breaker: smaller reposition time before)
                candidates.sort(key=lambda x: (x[1], x[2]['t_before_min']))
                chosen_bus, chosen_usage, fit = candidates[0]

                # Optional: only swap if chosen is actually less used than current bus
                if require_less_used and chosen_usage >= service_minutes.get(bus_id, float('inf')):
                    continue

                # Apply swap:
                # 1) Reassign the next_trip row to chosen_bus
                idx = next_trip['_orig_idx']
                df.loc[df['_orig_idx'] == idx, 'bus'] = chosen_bus

                # 2) Insert material trip BEFORE
                df = add_material_trip(
                    df, chosen_bus,
                    fit['prev_loc'],
                    next_trip['start location'],
                    fit['depart_time'],
                    next_trip['start time'],
                    fit['e_before']
                )

                # 3) Insert material trip AFTER (only if there's a known next event)
                if fit['t_after_min'] > 0 and fit['next_start_loc'] is not None:
                    df = add_material_trip(
                        df, chosen_bus,
                        next_trip['end location'],
                        fit['next_start_loc'],
                        next_trip['end time'],
                        next_trip['end time'] + datetime.timedelta(minutes=fit['t_after_min']),
                        fit['e_after']
                    )

                # Update "less used" metric locally (chosen bus gets more service minutes)
                trip_minutes = (next_trip['end time'] - next_trip['start time']).total_seconds() / 60
                service_minutes[chosen_bus] = service_minutes.get(chosen_bus, 0.0) + trip_minutes
                service_minutes[bus_id] = service_minutes.get(bus_id, 0.0) - trip_minutes

                # Restart the outer loops to use the updated df consistently
                df = df.sort_values('start time').reset_index(drop=True)
                changed = True
                break

            if changed:
                break

    # Drop helper column before returning
    if '_orig_idx' in df.columns:
        df = df.drop(columns=['_orig_idx'])

    return df.sort_values('start time').reset_index(drop=True)

def main():
    planning = pd.read_excel('../Excel Files/Bus Planning.xlsx')
    distance_matrix = pd.read_excel('../Excel Files/DistanceMatrix.xlsx')
    planning = rename_time_object(planning, 'start time', 'end time')

    # 1) Let a less-used bus take the next duty if it can reach it without breaking its own schedule.
    # - prefer_garage_only=True to only swap with buses that were last at the garage.
    # - require_less_used=True ensures swaps only when the replacement is indeed less utilized.
    planning_swapped = swap_buses_for_charging(
        planning, distance_matrix,
        prefer_garage_only=False,
        buffer_minutes=3,
        require_less_used=True
    )

    # 2) With swaps in place, your existing function can now insert charging/idle windows.
    updated_planning = find_insert_charging_and_idle_records(planning_swapped, distance_matrix)

    print("\nUpdated Planning (with Swaps + Charging/Idle):")
    print(updated_planning)

    updated_planning.to_excel("../Excel Files/UpdatedPlanningWithChargingIdleAndSwaps.xlsx", index=False)

if __name__ == "__main__":
    main()