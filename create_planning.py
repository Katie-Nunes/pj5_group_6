import pandas as pd
import datetime
from check_inaccuracies import rename_time_object, validate_dataframe_structure


def lookup_distance_matrix(start, end, line, distance_matrix_df):
    if start == 'ehvgar' or end == 'ehvgar':
        # For garage routes, don't filter by line
        row = distance_matrix_df[
            (distance_matrix_df['start'] == start) &
            (distance_matrix_df['end'] == end)
            ]
    else:
        # For regular routes, filter by line as well
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

    # Create a "material trip" record
    material_trip = [
        'ehvgar', location, departure_time.strftime('%H:%M:%S'), arrival_time.strftime('%H:%M:%S'),
        "material trip", line, distance_m / 1000, next_bus_id
    ]
    return material_trip

def initialize_bus_locations(timetable_df, distance_matrix_df, bus_locations, bus_status, next_bus_id, discard="ehvgar"):
    locations = set(timetable_df['start']).union(set(timetable_df['end']))
    locations.discard(discard)  # Don't need to worry about the garage

    for location in locations:
        count = 0
        for bus_id, loc in bus_locations.items():
            if loc == location:
                count += 1

        while count < 2:
            # Use the earliest departure time for the first bus at each location
            earliest_departure_time = timetable_df['departure_time'].min()
            line = timetable_df[timetable_df['start'] == location]['line'].iloc[0] #Get the line for this location

            material_trip = get_bus_from_garage(location, line, earliest_departure_time, distance_matrix_df, next_bus_id)
    return material_trip

def find_idle_bus_at_location(start_location, bus_locations, generated_data):
    bus_last_activity = {}
    for record in generated_data:
        bus_id = record[7]  # bus_id is at index 7
        bus_last_activity[bus_id] = record[4]  # activity is at index 4

    # Find a bus that is at start_location and whose last activity was "idle"
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
    # Calculate required garage departure time
    max_travel_from_garage, _, _ = lookup_distance_matrix('ehvgar', start_location, line, DISTANCEMATRIX)
    required_garage_departure = departure_time - datetime.timedelta(minutes=max_travel_from_garage)

    # Generate MATERIAL TRIP
    material_trip = get_bus_from_garage(start_location, line, required_garage_departure,
                                        DISTANCEMATRIX, next_bus_id)
    new_bus_id = next_bus_id
    generated_data.append(material_trip)
    bus_locations[new_bus_id] = start_location

    # Parse material trip end time and create IDLE record
    material_trip_end_time = datetime.datetime.strptime(material_trip[3], '%H:%M:%S')
    material_trip_end_time = departure_time.replace(
        hour=material_trip_end_time.hour,
        minute=material_trip_end_time.minute,
        second=material_trip_end_time.second
    )

    idle_record = create_idle_record(start_location, material_trip_end_time,
                                     departure_time, line, new_bus_id)
    generated_data.append(idle_record)

    # Create SERVICE TRIP with new bus
    service_trip, _ = create_service_trip(start_location, end_location, departure_time,
                                          line, new_bus_id, DISTANCEMATRIX)
    bus_locations[new_bus_id] = end_location
    generated_data.append(service_trip)

    return next_bus_id + 1


def main_timetable_iteration(TIMETABLE, DISTANCEMATRIX, bus_locations, generated_data, next_bus_id):
    for idx, row in TIMETABLE.iterrows():
        start_location = row['start']
        departure_time = row['departure_time']
        end_location = row['end']
        line = row['line']

        # Check for idle bus at start_location
        available_bus = find_idle_bus_at_location(start_location, bus_locations, generated_data)

        if available_bus:
            # Use existing idle bus
            handle_existing_bus(start_location, end_location, departure_time, line,
                                available_bus, bus_locations, generated_data, DISTANCEMATRIX)
        else:
            # Fetch new bus from garage
            next_bus_id = handle_new_bus(start_location, end_location, departure_time, line,
                                         next_bus_id, bus_locations, generated_data, DISTANCEMATRIX)

    return next_bus_id

def create_dataframe(generated_data):
    columns = ["start location", "end location", "start time", "end time", "activity", "line", "energy consumption", "bus"]
    generated_df = pd.DataFrame(generated_data, columns=columns)
    validate_dataframe_structure(generated_df, apply=True)
    return generated_df

def create_planning(TIMETABLE, DISTANCEMATRIX):
    TIMETABLE = rename_time_object(TIMETABLE, "departure_time", "Not Inside")

    bus_locations = {}
    generated_data = []
    next_bus_id = 1
    next_bus_id = main_timetable_iteration(TIMETABLE, DISTANCEMATRIX, bus_locations, generated_data, next_bus_id)
    output_df = create_dataframe(generated_data)
    return output_df

if "__name__" == "__main__":
    create_planning()



