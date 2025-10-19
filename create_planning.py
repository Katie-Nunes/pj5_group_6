import pandas as pd
import datetime
from check_inaccuracies import rename_time_object, validate_dataframe_structure
from check_feasbility import check_energy_feasibility

def create_service_record(start_location, end_location, departure_time, line, bus_id, distance_matrix, current_energy):
    travel_time, energy_use = lookup_distance_matrix(start_location, end_location, line, distance_matrix)
    arrival_time = departure_time + datetime.timedelta(minutes=travel_time)

    record = [
        start_location,
        end_location,
        departure_time.strftime('%H:%M:%S'),
        arrival_time.strftime('%H:%M:%S'),
        "service trip",
        line,
        current_energy - energy_use,
        bus_id,
    ]
    return record

def create_idle_record(location, start_time, end_time, line, bus_id, current_energy):
    delta = (end_time - start_time).total_seconds() / 3600.0
    idle_cost = delta * 5
    ending_energy = max(0.0, current_energy - idle_cost)  # Clamp to >=0

    record = [
        location,
        location,
        start_time.strftime('%H:%M:%S'),
        (end_time.strftime('%H:%M:%S') if end_time else None),
        "idle",
        line,
        ending_energy,
        bus_id,
    ]
    return record

def create_charging_record(start_time, charging_time, line, bus_id, current_energy, charge_speed_assumed=60):
    """Create a record for a charging session."""
    end_time = start_time + datetime.timedelta(minutes=charging_time)
    charging_amount = charging_time * charge_speed_assumed

    record = [
        "ehvgar",
        "ehvgar",
        start_time.strftime('%H:%M:%S'),
        end_time.strftime('%H:%M:%S'),
        "charging",
        line,
        current_energy + charging_amount,
        bus_id,
    ]
    return record

def create_material_record(destination, line, departure_time, distance_matrix, bus_id, current_energy):
    start_location = "ehvgar"
    travel_time, energy_use = lookup_distance_matrix(start_location, destination, line, distance_matrix)
    arrival_time = departure_time + datetime.timedelta(minutes=travel_time)

    record = [
        start_location,
        destination,
        departure_time.strftime('%H:%M:%S'),
        arrival_time.strftime('%H:%M:%S'),
        "material trip",
        line,
        current_energy - energy_use,
        bus_id,
    ]
    return record

def lookup_distance_matrix(start, end, line, distance_matrix_df):
    if start == 'ehvgar' or end == 'ehvgar':
        row = distance_matrix_df[
            (distance_matrix_df['start'] == start) &
            (distance_matrix_df['end'] == end)
        ]
    else:
        row = distance_matrix_df[
            (distance_matrix_df['start'] == start) &
            (distance_matrix_df['end'] == end) &
            (distance_matrix_df['line'] == line)
        ]
    max_travel_time = int(row.iloc[0]['max_travel_time'])
    distance_km = (int(row.iloc[0]['distance_m'])/1000)
    energy_use = distance_km*1.6
    return max_travel_time, energy_use








def initialize_buses(timetable, distance_matrix, next_bus_id=1, home="ehvgar"):
    """Ensure each location has a bus, generating simple material trips if needed."""
    locations = set(timetable['start']).union(timetable['end'])
    locations.discard(home)

    trips, bus_locations = [], {}
    earliest_departure = timetable['departure_time'].min()

    for location in locations:
        line = timetable.loc[timetable['start'] == location, 'line'].iloc[0]
        trip = create_material_record(location, line, earliest_departure, distance_matrix, next_bus_id)
        trips.append(trip)
        bus_locations[next_bus_id] = location
        next_bus_id += 1
    return trips, bus_locations, next_bus_id


def assign_bus(row, bus_locations, distance_matrix, generated_data, next_bus_id):
    """Either reuse an idle bus or fetch one from the garage."""
    start, end, dep, line = row['start'], row['end'], row['departure_time'], row['line']

    # find any idle bus already at the start location
    for bus_id, loc in bus_locations.items():
        if loc == start:
            trip = create_service_record(start, end, dep, line, bus_id, distance_matrix)
            generated_data.append(trip)
            bus_locations[bus_id] = end
            return next_bus_id

    # otherwise, bring a fresh bus from the garage
    travel_time, _, _ = lookup_distance_matrix("ehvgar", start, line, distance_matrix)
    garage_depart = dep - datetime.timedelta(minutes=travel_time)

    material_trip = create_material_record(start, line, garage_depart, distance_matrix, next_bus_id)
    generated_data.append(material_trip)

    service_trip = create_service_record(start, end, dep, line, next_bus_id, distance_matrix)
    generated_data.append(service_trip)

    bus_locations[next_bus_id] = end
    return next_bus_id + 1


def main_timetable_iteration(timetable, distance_matrix, next_bus_id=1):
    """Single‑pass timetable processing — simplified bus planning."""
    generated_data = []

    # initialize at least one bus per active location
    init_trips, bus_locations, next_bus_id = initialize_buses(timetable, distance_matrix, next_bus_id)
    generated_data.extend(init_trips)

    # create service trips for each timetable entry
    for _, row in timetable.iterrows():
        next_bus_id = assign_bus(row, bus_locations, distance_matrix, generated_data, next_bus_id)
    return generated_data, next_bus_id






def create_planning(timetable, distance_matrix):
    timetable = (rename_time_object(timetable, "departure_time", "Not Inside").sort_values(by="departure_time"))
    generated_data, _ = main_timetable_iteration(timetable, distance_matrix)

    df = pd.DataFrame(generated_data, columns=["start location", "end location", "start time", "end time","activity", "line", "energy consumption", "bus"])
    validate_dataframe_structure(df, apply=True)
    df.to_excel("Excel Files/CREATED.xlsx", index=False)
    return df