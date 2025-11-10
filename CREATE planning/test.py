

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

