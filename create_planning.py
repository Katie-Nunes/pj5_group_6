import pandas as pd
import datetime
from check_inaccuracies import rename_time_object, validate_dataframe_structure

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
    distance_m = int(row.iloc[0]['distance_m'])
    return max_travel_time, distance_m, 0

def get_bus_from_garage(location, line, departure_time, distance_matrix_df, next_bus_id):
    travel_time, distance_m, _ = lookup_distance_matrix('ehvgar', location, line, distance_matrix_df)
    arrival_time = departure_time + datetime.timedelta(minutes=travel_time)
    material_trip = [
        'ehvgar', location, departure_time.strftime('%H:%M:%S'), arrival_time.strftime('%H:%M:%S'),
        "material trip", line, distance_m / 1000, next_bus_id
    ]
    return material_trip


def initialize_bus_locations(timetable_df, distance_matrix_df, bus_locations, bus_status, next_bus_id, discard="ehvgar"):
    locations = set(timetable_df['start']).union(set(timetable_df['end']))
    locations.discard(discard)

    generated_trips = []
    current_bus_id = next_bus_id

    for location in locations:
        count = sum(1 for loc in bus_locations.values() if loc == location)
        while count < 2:
            earliest_departure_time = timetable_df['departure_time'].min()
            line = timetable_df[timetable_df['start'] == location]['line'].iloc[0]
            material_trip = get_bus_from_garage(location, line, earliest_departure_time, distance_matrix_df, current_bus_id)
            generated_trips.append(material_trip)
            bus_locations[current_bus_id] = location
            current_bus_id += 1
            count += 1

    return generated_trips, current_bus_id

def find_idle_bus_at_location(start_location, bus_locations, generated_data):
    bus_last_activity = {record[7]: record[4] for record in generated_data}
    for bus_id, location in bus_locations.items():
        if location == start_location and bus_last_activity.get(bus_id, "idle") == "idle":
            return bus_id
    return None


def create_service_trip(start_location, end_location, departure_time, line, bus_id, DISTANCEMATRIX):
    travel_time, distance_m, _ = lookup_distance_matrix(start_location, end_location, line, DISTANCEMATRIX)
    arrival_time = departure_time + datetime.timedelta(minutes=travel_time)
    service_trip = [
        start_location,
        end_location,
        departure_time.strftime('%H:%M:%S'),
        arrival_time.strftime('%H:%M:%S'),
        "service trip",
        line,
        distance_m / 1000,
        bus_id
    ]
    return service_trip, arrival_time


def create_idle_record(location, start_time, end_time, line, bus_id):
    return [
        location,
        location,
        start_time.strftime('%H:%M:%S'),
        end_time.strftime('%H:%M:%S'),
        "idle",
        line,
        0,
        bus_id
    ]


def handle_existing_bus(start_location, end_location, departure_time, line, available_bus, bus_locations, generated_data, DISTANCEMATRIX):
    service_trip, _ = create_service_trip(start_location, end_location, departure_time, line, available_bus, DISTANCEMATRIX)
    bus_locations[available_bus] = end_location
    generated_data.append(service_trip)


def handle_new_bus(start_location, end_location, departure_time, line, next_bus_id, bus_locations, generated_data, DISTANCEMATRIX):
    max_travel_from_garage, _, _ = lookup_distance_matrix('ehvgar', start_location, line, DISTANCEMATRIX)
    required_garage_departure = departure_time - datetime.timedelta(minutes=max_travel_from_garage)
    material_trip = get_bus_from_garage(start_location, line, required_garage_departure, DISTANCEMATRIX, next_bus_id)
    new_bus_id = next_bus_id
    generated_data.append(material_trip)
    bus_locations[new_bus_id] = start_location

    material_trip_end_time_str = material_trip[3]
    material_trip_end_time = datetime.datetime.strptime(material_trip_end_time_str, '%H:%M:%S')
    material_trip_end_time = departure_time.replace(
        hour=material_trip_end_time.hour,
        minute=material_trip_end_time.minute,
        second=material_trip_end_time.second
    )

    idle_record = create_idle_record(start_location, material_trip_end_time, departure_time, line, new_bus_id)
    generated_data.append(idle_record)

    service_trip, _ = create_service_trip(start_location, end_location, departure_time, line, new_bus_id, DISTANCEMATRIX)
    bus_locations[new_bus_id] = end_location
    generated_data.append(service_trip)

    return next_bus_id + 1


def main_timetable_iteration(TIMETABLE, DISTANCEMATRIX, bus_locations, generated_data, next_bus_id):
    for idx, row in TIMETABLE.iterrows():
        start_location = row['start']
        departure_time = row['departure_time']
        end_location = row['end']
        line = row['line']

        available_bus = find_idle_bus_at_location(start_location, bus_locations, generated_data)

        if available_bus:
            handle_existing_bus(start_location, end_location, departure_time, line, available_bus, bus_locations, generated_data, DISTANCEMATRIX)
        else:
            next_bus_id = handle_new_bus(start_location, end_location, departure_time, line, next_bus_id, bus_locations, generated_data, DISTANCEMATRIX)

    return next_bus_id


def create_dataframe(generated_data):
    columns = ["start location", "end location", "start time", "end time", "activity", "line", "energy consumption", "bus"]
    generated_df = pd.DataFrame(generated_data, columns=columns)
    validate_dataframe_structure(generated_df, apply=True)
    return generated_df


def create_planning(TIMETABLE, DISTANCEMATRIX):
    TIMETABLE = rename_time_object(TIMETABLE, "departure_time", "Not Inside")
    TIMETABLE = TIMETABLE.sort_values(by='departure_time', ascending=True)

    bus_locations = {}
    generated_data = []
    next_bus_id = 1
    next_bus_id = main_timetable_iteration(TIMETABLE, DISTANCEMATRIX, bus_locations, generated_data, next_bus_id)
    output_df = create_dataframe(generated_data)
    output_df.to_excel("Excel Files/CREATED.xlsx")
    return output_df


if __name__ == "__main__":
    create_planning()