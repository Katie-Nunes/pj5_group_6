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

def create_material_record(start_location, end_location, line, departure_time, distance_matrix, bus_id, current_energy):
    travel_time, energy_use = lookup_distance_matrix(start_location, end_location, line, distance_matrix)
    arrival_time = departure_time + datetime.timedelta(minutes=travel_time)

    record = [
        start_location,
        end_location,
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

def create_garage_trips(timetable, distance_matrix, row, bus_id, current_energy, togarage=False):
    locations = {loc: {} for loc in set(timetable['start']).union(timetable['end'])}
    row_values = {}
    for column in ["start", "departure_time", "end", "line"]:
        row_values[column] = timetable.at[row, column]

    if togarage:
        record = create_material_record(row_values["start"], 'ehvgar', row_values["line"], row_values["departure_time"], distance_matrix, bus_id, current_energy)
        locations.setdefault("ehvgar", []).append(bus_id) # this is kinda iffy, maybe change method of doing this later
        return record

    else: #From garage to whatever the hell the current location is
        if "ehvgar" in locations and locations["ehvgar"] is False: # Check if value in ehvgar dict is empty list (no busses at garage)
            new_bus_id = max(list(locations.values())) + 1 #get highest then plus 1
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, new_bus_id, 255) # new bus always start at max of SoH
            locations[row_values["start"]].append(new_bus_id)
            return record
        else: # there is already at least one bus at garage
            garage_busses = locations.get("ehvgar")
            for busses in garage_busses:
                # for every bus in there check battery level, pick highest one which also fulfills this
                # Check bus battery (Go back in time and see bus battery needs to leave is >60% (arbitrary) )
                pass  # return best bus

                if False:  # Yes, battery above 60%
                    pass
                else:
                    new_bus_id = max(list(locations.values())) + 1 #get highest then plus 1
                    record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, new_bus_id, 255) # new bus always start at max of SoH
                    locations[row_values["start"]].append(new_bus_id)
                    return record
            pass

        record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, bus_id, current_energy)
        return record





def main_timetable_iteration(timetable, distance_matrix):
    pass

def create_planning(timetable, distance_matrix):
    timetable = (rename_time_object(timetable, "departure_time", "Not Inside").sort_values(by="departure_time"))
    timetable.columns = [col.replace(' ', '_').lower() for col in timetable.columns]
    generated_data, _ = main_timetable_iteration(timetable, distance_matrix)

    df = pd.DataFrame(generated_data, columns=["start location", "end location", "start time", "end time","activity", "line", "energy consumption", "bus"])
    validate_dataframe_structure(df, apply=True)
    df.to_excel("Excel Files/CREATED.xlsx", index=False)
    return df