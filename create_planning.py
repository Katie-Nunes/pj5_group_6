import pandas as pd
import numpy as np
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
    locations = {loc: [] for loc in set(timetable['start']).union(timetable['end'])}
    row_values = {}
    chargingz = {}

    # Get just the current values from the timetable row
    for column in ["start", "departure_time", "end", "line"]:
        row_values[column] = timetable.at[row, column]

    if togarage:
        if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
            locations[row_values["start"]].remove(bus_id)
        locations.setdefault("ehvgar", []).append(bus_id)  # this is kinda iffy, maybe change method of doing this later
        record = create_material_record(row_values["start"], 'ehvgar', row_values["line"], row_values["departure_time"], distance_matrix, bus_id, current_energy)
        chargingz[bus_id] = chargingz.get(bus_id, [])  # Initialize if not present
        chargingz[bus_id].append({"arrival_time": record[3], "ze_energy": current_energy})  # Iloc to 4th item inside the record list, end time
        return record

    else:
        if not locations.get("ehvgar", []):
            if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
                locations[row_values["start"]].remove(bus_id)

            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            locations[row_values["start"]].append(new_bus_id)
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, new_bus_id, 255)  # new bus always start at max of SoH
            return record

        # If garage has buses: evaluate viable options
        garage_buses = locations["ehvgar"]
        viable_bus_ids = {}

        for bus_id in garage_buses:
            # Get charging records for the bus (safe fallback if no records exist)
            bus_charging_records = chargingz.get(bus_id, [])
            if not bus_charging_records:
                continue  # Skip if bus has no charging history

            most_recent_arrival = max(item["arrival_time"] for item in chargingz[bus])  # get the most recent item of the specific bus of the list of datetime objects inside charginz
            travel_gar_to_loc, _ = lookup_distance_matrix('ehvgar', row_values["departure_time"], row_values["line"], distance_matrix)
            time_charging = row_values["departure_time"] - most_recent_arrival - datetime.timedelta(minutes=travel_gar_to_loc)  # Calculated the total time the bus would spend charging assuming it leaves early to be on departure on time
            time_charging_hours = time_charging.total_seconds() / 3600
            charged_amount = time_charging_hours * 450
            bus_pre_depart_energy = chargingz[bus_id][-1]["ze_energy"] + charged_amount  # Use the last recorded energy level

            if bus_pre_depart_energy >= 153:
                viable_bus_ids[bus_id] = bus_pre_depart_energy

        if not viable_bus_ids:
            # No viable buses: create new one
            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            locations[row_values["start"]].append(new_bus_id)
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, new_bus_id, 255)  # new bus always start at max of SoH
        else:
            best_bus = max(viable_bus_ids, key=viable_bus_ids.get)
            locations["ehvgar"].remove(best_bus)
            locations[row_values["start"]].append(best_bus)
            bus_pre_depart_energy = viable_bus_ids[best_bus]
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, best_bus, bus_pre_depart_energy)
        return record





def main_timetable_iteration(timetable, distance_matrix):
    pass

def create_planning(timetable, distance_matrix):
    timetable = (rename_time_object(timetable, "departure_time", "Not Inside").sort_values(by="departure_time"))
    timetable.columns = [col.replace(' ', '_').lower() for col in timetable.columns]
    generated_data = main_timetable_iteration(timetable, distance_matrix)

    df = pd.DataFrame(generated_data, columns=["start location", "end location", "start time", "end time","activity", "line", "energy consumption", "bus"])
    validate_dataframe_structure(df, apply=True)
    df.to_excel("Excel Files/CREATED.xlsx", index=False)
    return df