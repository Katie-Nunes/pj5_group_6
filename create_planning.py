import pandas as pd
import datetime
from datetime import date
import numpy as np
from check_inaccuracies import rename_time_object, validate_dataframe_structure
#from check_feasibility import check_energy_feasibility

## For now I'm misusing energy consumption column, the deltas between previous and current can be calculated post hoc and edited2

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

def create_idle_record(location, start_time, end_time, bus_id, current_energy):
    delta = (end_time - start_time).total_seconds() / 3600.0
    idle_cost = delta * 5
    ending_energy = max(0.0, current_energy - idle_cost)  # Clamp to >=0

    record = [
        location,
        location,
        start_time.strftime('%H:%M:%S'),
        (end_time.strftime('%H:%M:%S') if end_time else None),
        "idle",
        "999",
        ending_energy,
        bus_id,
    ]
    return record

def create_charging_record(start_time, charging_time, line, bus_id, current_energy, charge_speed_assumed=450):
    """Create a record for a charging session."""
    end_time = start_time + datetime.timedelta(minutes=charging_time)
    charging_amount = charging_time * charge_speed_assumed
    new_current_energy = current_energy + charging_amount
    min(new_current_energy, 229.5) # cap at 90% to not have to deal with slow charging
    record = [
        "ehvgar",
        "ehvgar",
        start_time.strftime('%H:%M:%S'),
        end_time.strftime('%H:%M:%S'),
        "charging",
        line,
        new_current_energy,
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
        999,
        current_energy - energy_use,
        bus_id,
    ]
    return record


def lookup_distance_matrix(start, end, line, distance_matrix_df):
    # Determine the line to use based on garage condition
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

    if row.empty:
        # Provide detailed error context for debugging
        raise ValueError(
            f"Missing distance/time entry for: "
            f"start={start}, end={end}, line={line}. "
            f"Check distance_matrix_df for these parameters."
        )

    max_travel_time = int(row.iloc[0]['max_travel_time'])
    distance_km = (int(row.iloc[0]['distance_m']) / 1000)
    energy_use = distance_km * 1.6
    return max_travel_time, energy_use

def create_garage_trips(timetable, distance_matrix, index,  locations, charging_dict, bus_id=None, current_energy=None):

    row_values = {}
    # Get just the current values from the timetable row
    for column in ["start", "departure_time", "end", "line"]:
        row_values[column] = timetable.at[index, column]

    if bus_id is not None: # we assume that if you give a bus id, then you want to go to the garage
        if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
            locations[row_values["start"]].remove(bus_id)

        locations.setdefault("ehvgar", []).append(bus_id)  # this is kinda iffy, maybe change method of doing this later
        record = create_material_record(row_values["start"], 'ehvgar', row_values["line"], row_values["departure_time"], distance_matrix, bus_id, current_energy)
        charging_dict[bus_id] = charging_dict.get(bus_id, [])  # Initialize if not present
        charging_dict[bus_id].append({"arrival_time": record[3], "ze_energy": current_energy})  # iloc to 4th item inside the record list, end time
        return record

    else:
        if not locations.get("ehvgar", []): # Garage doesn't have any buses
            if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
                locations[row_values["start"]].remove(bus_id)

            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            locations[row_values["start"]].append(new_bus_id)
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, new_bus_id, 255)  # new bus always start at max of SoH
            return record, None

        # If garage has buses: evaluate viable options
        garage_buses = locations["ehvgar"]
        viable_bus_ids = {}

        for bus_id in garage_buses:
            # Get charging records for the bus (safe fallback if no records exist)
            bus_charging_records = charging_dict.get(bus_id, [])
            if not bus_charging_records:
                continue  # Skip if bus has no charging history

            most_recent_arrival_str = max(item["arrival_time"] for item in charging_dict[bus_id])  # get the most recent item of the specific bus of the list of datetime objects inside charging_dict
            most_recent_arrival = datetime.datetime.combine(date.today(), datetime.datetime.strptime(most_recent_arrival_str, "%H:%M:%S").time())
            travel_gar_to_loc, _ = lookup_distance_matrix('ehvgar', row_values["end"], row_values["line"], distance_matrix)
            time_charging = row_values["departure_time"] - most_recent_arrival - datetime.timedelta(minutes=travel_gar_to_loc)  # Calculated the total time the bus would spend charging assuming it leaves early to be on departure on time
            time_charging_hours = time_charging.total_seconds() / 3600
            charged_amount = time_charging_hours * 450
            bus_pre_depart_energy = charging_dict[bus_id][-1]["ze_energy"] + charged_amount  # Use the last recorded energy level

            if bus_pre_depart_energy >= 153:
                viable_bus_ids[bus_id] = bus_pre_depart_energy

        if not viable_bus_ids:
            # No viable buses: create new one
            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            locations[row_values["start"]].append(new_bus_id)
            charging_record = None
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, new_bus_id, 255)  # new bus always start at max of SoH
        else:
            best_bus = max(viable_bus_ids, key=viable_bus_ids.get)
            charging_record = create_charging_record(most_recent_arrival, (time_charging_hours/60), "999", best_bus, bus_pre_depart_energy)

            locations["ehvgar"].remove(best_bus)
            locations[row_values["start"]].append(best_bus)

            bus_pre_depart_energy = viable_bus_ids[best_bus]
            record = create_material_record('ehvgar', row_values["start"], row_values["line"], row_values["departure_time"], distance_matrix, best_bus, bus_pre_depart_energy)
        return record, charging_record

def main_timetable_iteration(timetable, distance_matrix, locations, charging_dict):
    generated_data = pd.DataFrame(columns=["start location", "end location", "start time", "end time", "activity", "line","energy consumption", "bus"])
    for index, row in timetable.iterrows():
        timetable_cols = ["start", "departure_time", "end", "line"]
        row_vals = dict.fromkeys(timetable_cols)  # Fixed: was list, can't assign string keys to lists

        if not generated_data.empty:
            bus_id = generated_data["bus"].iloc[-1]
            current_energy = generated_data["energy consumption"].iloc[-1]  # Currently this does not handle multiple bus-lines in timetable or the first line not existing
        else:
            bus_id = 1
            current_energy = 255

        for column in timetable_cols:
            row_vals[column] = timetable.at[index, column]

        busses_at_current_location = locations[row_vals["start"]]
        if not busses_at_current_location:
            pre_trip, charging_trip = create_garage_trips(timetable, distance_matrix, index, locations, charging_dict)

            trip = create_service_record(row_vals["start"], row_vals["end"], row_vals["departure_time"], row_vals["line"], bus_id, distance_matrix, current_energy, )
            if charging_trip:
                generated_data = pd.concat([
                    generated_data,
                    pd.DataFrame([charging_trip], columns=generated_data.columns),
                    pd.DataFrame([pre_trip], columns=generated_data.columns),
                    pd.DataFrame([trip], columns=generated_data.columns)
                ], ignore_index=True)
            else:
                generated_data = pd.concat([
                    generated_data,
                    pd.DataFrame([pre_trip], columns=generated_data.columns),
                    pd.DataFrame([trip], columns=generated_data.columns)
                ], ignore_index=True)

        else:
            if current_energy > 25.5: #10%
                # No need for idle since that gets fixed in post by other script. create_idle_record(row_vals["start"], start_time, row_vals["departure_time"], bus_id, current_energy) # if a bus is never used again it doesn't get an idling record, ditto with charging
                current_energy = generated_data["energy consumption"].iloc[-1]
                trip = create_service_record(row_vals["start"], row_vals["end"], row_vals["departure_time"], row_vals["line"], bus_id, distance_matrix, current_energy, )
                generated_data = pd.concat([generated_data, pd.DataFrame([pre_trip], columns=generated_data.columns)], ignore_index=True)
            else:
                send_to_garage = create_garage_trips(timetable, distance_matrix, index, locations, charging_dict, bus_id, current_energy) # Assumes energy doesn't run out while idling
                pre_trip, _ = create_garage_trips(timetable, distance_matrix, index, locations, charging_dict)
                generated_data = pd.concat([
                    generated_data,
                    pd.DataFrame([send_to_garage], columns=generated_data.columns),
                    pd.DataFrame([pre_trip], columns=generated_data.columns)
                ], ignore_index=True)
    return generated_data



def create_planning(timetable, distance_matrix):
    locations = {loc: [] for loc in set(timetable['start']).union(timetable['end'])}
    charging_dict = {}

    timetable = (rename_time_object(timetable, "departure_time", "Not Inside").sort_values(by="departure_time"))
    timetable.columns = [col.replace(' ', '_').lower() for col in timetable.columns]

    generated_data = main_timetable_iteration(timetable, distance_matrix, locations, charging_dict)
    validate_dataframe_structure(generated_data, apply=True)

    generated_data.to_excel("Excel Files/CREATED.xlsx", index=False)
    return generated_data


print("tits")