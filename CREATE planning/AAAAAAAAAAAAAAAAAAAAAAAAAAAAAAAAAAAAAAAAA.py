import pandas as pd
from datetime import datetime, timedelta
from check_inaccuracies import rename_time_object, validate_dataframe_structure

GARAGE = 'ehvgar'


def lookup_distance_matrix(start, end, line, distance_matrix_df):
    """
    Queries long-format distance matrix for travel time and energy.
    """
    print(f"    [LOOKUP] Querying distance matrix: {start} -> {end}, Line {line}")

    if start == GARAGE or end == GARAGE:
        print(f"      - Garage detected, ignoring line parameter")
        match = (distance_matrix_df['start'] == start) & (distance_matrix_df['end'] == end)
    else:
        match = (
            (distance_matrix_df['start'] == start) &
            (distance_matrix_df['end'] == end) &
            (distance_matrix_df['line'] == line)
        )

    row = distance_matrix_df[match]

    if row.empty:
        print(f"      - ERROR: No matching distance matrix entry found!")
        raise ValueError(
            f"Missing distance/time entry for: "
            f"start={start}, end={end}, line={line}. "
            f"Check distance_matrix_df for these parameters."
        )

    travel_time = int(row.iloc[0]['max_travel_time'])
    distance_km = int(row.iloc[0]['distance_m']) / 1000
    energy = distance_km * 1.6
    print(f"      - Found: {distance_km}km, {travel_time} mins, {energy:.2f} energy")

    return travel_time, energy


def create_service_record(start_loc, end_loc, departure_time, line, bus, dm_df):
    """
    Creates a record for passenger service trip and updates the bus state.
    """
    travel_time_mins, energy_used = lookup_distance_matrix(start_loc, end_loc, line, dm_df)

    bus['location'] = end_loc
    bus['energy'] -= energy_used
    bus['available_time'] = departure_time + timedelta(minutes=travel_time_mins)

    return {
        "start location": start_loc,
        "end location": end_loc,
        "start time": departure_time,
        "end time": bus['available_time'],
        "activity": "service trip",
        "line": line,
        "energy consumption": bus['energy'],  # after trip
        "bus": bus['id']
    }


def create_garage_trips(timetable, dm_df, trip_index, charging_dict, bus, departure_time):
    """
    Prepares a bus (deadhead + charge + position) for its next trip from 'ehvgar'.
    """
    records = []

    if bus['location'] != GARAGE:
        print(f"  - Bus {bus['id']} is not at the garage, performing material trip to garage")
        mins, en = lookup_distance_matrix(bus['location'], GARAGE, 999, dm_df)
        end_time = departure_time - timedelta(minutes=mins)

        records.append({
            "start location": bus['location'],
            "end location": GARAGE,
            "start time": bus['available_time'],
            "end time": end_time,
            "activity": "material trip",
            "line": 999,
            "energy consumption": bus['energy'] - en,
            "bus": bus['id']
        })

        bus['location'] = GARAGE
        bus['energy'] -= en
        bus['available_time'] = end_time

    if bus['energy'] < charging_dict.get('min_energy', 51):
        print(f"  - Charging required: Bus {bus['id']} has {bus['energy']:.2f} energy (min: {charging_dict['min_energy']})")
        charge_hours = (charging_dict['max_energy'] - bus['energy']) / charging_dict['charging_rate']
        start = bus['available_time']
        end = start + timedelta(hours=charge_hours)

        records.append({
            "start location": GARAGE,
            "end location": GARAGE,
            "start time": start,
            "end time": end,
            "activity": "Charging",
            "line": None,
            "energy consumption": charging_dict['max_energy'],
            "bus": bus['id']
        })

        bus['energy'] = charging_dict['max_energy']
        bus['available_time'] = end
        print(f"    ➤ Charged to full: {bus['energy']} energy, ready at {bus['available_time']}")

    start_loc = timetable.at[trip_index, 'start']
    print(f"  - Sending bus from garage to stop {start_loc}")
    mins, en = lookup_distance_matrix(GARAGE, start_loc, None, dm_df)
    latest_departure = departure_time - timedelta(minutes=mins)

    if bus['available_time'] > latest_departure:
        raise ValueError(
            f"Bus {bus['id']} cannot get from {GARAGE} to {start_loc} before {departure_time}"
        )

    records.append({
        "start location": GARAGE,
        "end location": start_loc,
        "start time": bus['available_time'],
        "end time": departure_time,
        "activity": "material trip",
        "line": 999,
        "energy consumption": bus['energy'] - en,
        "bus": bus['id']
    })

    bus['location'] = start_loc
    bus['energy'] -= en
    bus['available_time'] = departure_time

    return records


def main_timetable_iteration(timetable, dm_df, charging_dict, fleet_size=5):
    """
    Goes through all timetable rows, assigns trips to suitable buses.
    """
    print(f"[MAIN ITERATION] Starting timetable iteration with a fleet of {fleet_size} buses.")

    fleet = [{
        "id": i,
        "location": GARAGE,
        "energy": charging_dict['max_energy'],
        "available_time": datetime.min
    } for i in range(1, fleet_size + 1)]

    generated_data = []

    for index, row in timetable.iterrows():
        print(f"\n[ITERATION {index}] Processing trip: {row['start']} -> {row['end']} at {row['departure_time']}")

        required_start = row['start']
        departure = row['departure_time']
        selected_bus = None

        for bus in fleet:
            if bus['location'] == required_start and bus['available_time'] <= departure:
                selected_bus = bus
                print(f"  - Found available Bus {bus['id']} already at {required_start}.")
                break

        if not selected_bus:
            print(f"  - No bus available at {required_start}. Finding best bus from fleet.")
            sorted_fleet = sorted(fleet, key=lambda b: b['available_time'])

            if not sorted_fleet or sorted_fleet[0]['available_time'] > departure:
                raise RuntimeError(
                    f"No bus in the entire fleet is available by {departure}."
                )

            selected_bus = sorted_fleet[0]
            print(f"  - Selected Bus {selected_bus['id']} from {selected_bus['location']}. It will be deadheaded.")

            deadhead_records = create_garage_trips(
                timetable, dm_df, index, charging_dict, selected_bus, departure
            )

            if deadhead_records:
                print(f"  - Adding {len(deadhead_records)} deadhead/charging record(s) for Bus {selected_bus['id']}.")
                generated_data.extend(deadhead_records)

        _, energy_needed = lookup_distance_matrix(required_start, row['end'], row['line'], dm_df)
        if selected_bus['energy'] < energy_needed:
            print(
                f"  - WARNING: Bus {selected_bus['id']} has insufficient energy ({selected_bus['energy']:.2f}) "
                f"for trip ({energy_needed:.2f}). Forcing charge.")
            selected_bus['energy'] = charging_dict['max_energy']

        print(f"  - Assigning service trip to Bus {selected_bus['id']}. Current energy: {selected_bus['energy']:.2f}")
        service_record = create_service_record(
            required_start, row['end'], departure, row['line'], selected_bus, dm_df
        )

        generated_data.append(service_record)
        print(
            f"  - Bus {selected_bus['id']} now at {selected_bus['location']} with energy {selected_bus['energy']:.2f}, "
            f"available at {selected_bus['available_time']}.")

    print("\n[MAIN ITERATION] Completed timetable iteration")
    return pd.DataFrame(generated_data).sort_values(by='start time').reset_index(drop=True)


def main(timetable, distance_matrix):
    print("[CREATE PLANNING] Initializing planning process")

    charging_dict = {
        'min_energy': 51,
        'max_energy': 240,
        'charging_rate': 50  # units/hour
    }


    locations = {loc: [] for loc in set(timetable['start']).union(timetable['end'])}
    print(f"  - Initialized locations: {list(locations.keys())}")
    print(f"  - Initialized charging dictionary")

    print(f"  - Renaming and sorting timetable by departure_time")
    timetable = rename_time_object(timetable, "departure_time", "Not Inside")
    print(f"  - Timetable sorted, {len(timetable)} rows")

    print("  - Shifting departure times back by 1 hour")
    timetable['departure_time'] = timetable['departure_time'] - timedelta(hours=1)

    timetable.sort_values(by="departure_time")

    print(f"  - Starting main timetable iteration")
    generated_data = main_timetable_iteration(timetable, distance_matrix, charging_dict, fleet_size=20)

    print(f"  - Validating generated data structure")
    validate_dataframe_structure(generated_data, apply=True)

    # ⏩ Shift generated times forward again
    print("  - Shifting generated times forward by 1 hour before export")
    for col in ['start time', 'end time']:
        generated_data[col] = generated_data[col] + timedelta(hours=1)

    print(f"  - Writing output to Excel: Excel Files/CREATED.xlsx")
    generated_data.to_excel("../Excel Files/CREATED-1.xlsx", index=False)

    print(f"[CREATE PLANNING] Planning complete! Generated {len(generated_data)} records")
    return generated_data

if __name__ == "__main__":
    PLANNING = pd.read_excel('../Excel Files/Bus Planning.xlsx')
    TIMETABLE = pd.read_excel('../Excel Files/Timetable.xlsx')
    DISTANCEMATRIX = pd.read_excel('../Excel Files/DistanceMatrix.xlsx')
    main(TIMETABLE, DISTANCEMATRIX)