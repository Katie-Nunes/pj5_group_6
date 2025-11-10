import pandas as pd
import datetime
from datetime import date
from check_inaccuracies import rename_time_object, validate_dataframe_structure


# from check_feasibility import check_energy_feasibility

## For now I'm misusing energy consumption column, the deltas between previous and current can be calculated post hoc and edited2

def create_service_record(start_location, end_location, departure_time, line, bus_id, distance_matrix, current_energy):
    print(
        f"  [SERVICE RECORD] Creating service trip: {start_location} -> {end_location}, Bus {bus_id}, Energy: {current_energy}")
    travel_time, energy_use = lookup_distance_matrix(start_location, end_location, line, distance_matrix)
    print(f"    - Travel time: {travel_time} mins, Energy use: {energy_use}")
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
    print(f"    - Final energy after trip: {current_energy - energy_use}")
    return record


def create_idle_record(location, start_time, end_time, bus_id, current_energy):
    print(f"  [IDLE RECORD] Creating idle record at {location}, Bus {bus_id}")
    delta = (end_time - start_time).total_seconds() / 3600.0
    print(f"    - Idle duration: {delta} hours")
    idle_cost = delta * 5
    ending_energy = max(0.0, current_energy - idle_cost)  # Clamp to >=0
    print(f"    - Idle cost: {idle_cost}, Energy before: {current_energy}, after: {ending_energy}")

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
    print(f"  [CHARGING RECORD] Creating charging record, Bus {bus_id}, Duration: {charging_time} mins")
    end_time = start_time + datetime.timedelta(minutes=charging_time)
    charging_amount = charging_time * charge_speed_assumed
    print(f"    - Charge speed: {charge_speed_assumed} kWh/min, Total charge amount: {charging_amount}")
    new_current_energy = current_energy + charging_amount
    print(f"    - Energy before: {current_energy}, after charging (before cap): {new_current_energy}")
    new_new_current_energy = min([new_current_energy, 229.5])  # cap at 90% to not have to deal with slow charging
    print(f"    - Energy after capping at 229.5: {new_new_current_energy}")

    record = [
        "ehvgar",
        "ehvgar",
        start_time.strftime('%H:%M:%S'),
        end_time.strftime('%H:%M:%S'),
        "charging",
        line,
        new_new_current_energy,
        bus_id,
    ]
    return record


def create_material_record(start_location, end_location, line, departure_time, distance_matrix, bus_id, current_energy):
    print(
        f"  [MATERIAL RECORD] Creating material trip: {start_location} -> {end_location}, Bus {bus_id}, Energy: {current_energy}")
    travel_time, energy_use = lookup_distance_matrix(start_location, end_location, line, distance_matrix)
    print(f"    - Travel time: {travel_time} mins, Energy use: {energy_use}")
    arrival_time = departure_time + datetime.timedelta(minutes=travel_time)
    cc_energy = current_energy - energy_use
    print(f"    - Final energy: {cc_energy}")

    record = [
        start_location,
        end_location,
        departure_time.strftime('%H:%M:%S'),
        arrival_time.strftime('%H:%M:%S'),
        "material trip",
        999,
        cc_energy,
        bus_id,
    ]
    return record, cc_energy


def lookup_distance_matrix(start, end, line, distance_matrix_df):
    print(f"    [LOOKUP] Querying distance matrix: {start} -> {end}, Line {line}")
    # Determine the line to use based on garage condition
    if start == 'ehvgar' or end == 'ehvgar':
        print(f"      - Garage detected, ignoring line parameter")
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
        print(f"      - ERROR: No matching distance matrix entry found!")
        # Provide detailed error context for debugging
        raise ValueError(
            f"Missing distance/time entry for: "
            f"start={start}, end={end}, line={line}. "
            f"Check distance_matrix_df for these parameters."
        )

    max_travel_time = int(row.iloc[0]['max_travel_time'])
    distance_km = (int(row.iloc[0]['distance_m']) / 1000)
    energy_use = distance_km * 1.6
    print(f"      - Found: {distance_km}km, {max_travel_time} mins, {energy_use} energy")
    return max_travel_time, energy_use


def record_garage_arrival(charging_dict, bus_id, arrival_time_str, energy):
    """Utility – always store the latest arrival/energy for a bus."""
    print(f"    [GARAGE ARRIVAL] Recording arrival - Bus {bus_id}, Time: {arrival_time_str}, Energy: {energy}")
    charging_dict.setdefault(bus_id, []).append(
        {"arrival_time": arrival_time_str, "ze_energy": energy}
    )


def create_garage_trips(timetable, distance_matrix, index, locations, charging_dict, bus_id=None, current_energy=None):
    print(f"  [GARAGE TRIPS] Processing index {index}, bus_id: {bus_id}, current_energy: {current_energy}")

    row_values = {}
    # Get just the current values from the timetable row
    for column in ["start", "departure_time", "end", "line"]:
        row_values[column] = timetable.at[index, column]

    print(
        f"    - Row values: start={row_values['start']}, departure={row_values['departure_time']}, end={row_values['end']}, line={row_values['line']}")

    if bus_id is not None:  # we assume that if you give a bus id, then you want to go to the garage
        print(f"    - Bus ID provided, sending to garage")
        if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
            locations[row_values["start"]].remove(bus_id)
            print(f"      - Removed Bus {bus_id} from {row_values['start']}")

        locations.setdefault("ehvgar", []).append(bus_id)  # this is kinda iffy, maybe change method of doing this later
        print(f"      - Added Bus {bus_id} to garage")
        record, cur_energy = create_material_record(row_values["start"], 'ehvgar', row_values["line"],
                                                    row_values["departure_time"], distance_matrix, bus_id,
                                                    current_energy)
        # Inside create_garage_trips, after you have built `record`:
        if row_values["end"] == "ehvgar":  # <-- you know the trip ends at the garage
            print(f"      - Trip ends at garage, recording arrival")
            record_garage_arrival(charging_dict, bus_id, record[3], cur_energy)
            if bus_id in locations[row_values["start"]]:
                locations[row_values["start"]].remove(bus_id)
            locations.setdefault("ehvgar", []).append(bus_id)
        return record, None

    else:
        print(f"    - No bus ID provided, checking garage status")
        if not locations.get("ehvgar", []):  # Garage doesn't have any buses
            print(f"      - Garage empty, creating new bus")
            if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
                locations[row_values["start"]].remove(bus_id)

            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            print(f"      - New bus ID: {new_bus_id}")
            locations[row_values["start"]].append(new_bus_id)
            max_tt, _ = lookup_distance_matrix('ehvgar', row_values["start"], row_values["line"], distance_matrix)
            pre_trip_depart = row_values["departure_time"] - datetime.timedelta(minutes=max_tt) # - max_time taken in timedelta
            record, _ = create_material_record('ehvgar', row_values["start"], row_values["line"], pre_trip_depart,
                                               distance_matrix, new_bus_id, 255)  # new bus always start at max of SoH
            return record, None

        # If garage has buses: evaluate viable options
        print(f"    - Garage has buses, evaluating viability")
        garage_buses = locations["ehvgar"]
        print(f"      - Buses in garage: {garage_buses}")
        viable_bus_ids = {}
        q = False  # Initialize ONCE before loop

        for bus_id in garage_buses:
            print(f"      - Evaluating Bus {bus_id}")
            # Get charging records for the bus (safe fallback if no records exist)
            bus_charging_records = charging_dict.get(bus_id, [])
            if not bus_charging_records:
                print(f"        - No charging records for Bus {bus_id}, skipping")
                continue  # Skip if bus has no charging history

            most_recent_arrival_str = max(item["arrival_time"] for item in charging_dict[
                bus_id])  # get the most recent item of the specific bus of the list of datetime objects inside charging_dict
            print(f"        - Most recent arrival: {most_recent_arrival_str}")
            most_recent_arrival_dt = datetime.datetime.combine(date.today(),
                                                               datetime.datetime.strptime(most_recent_arrival_str,
                                                                                          "%H:%M:%S").time())
            most_recent_arrival_final = pd.Timestamp(most_recent_arrival_dt)

            travel_gar_to_loc, _ = lookup_distance_matrix('ehvgar', row_values["end"], row_values["line"],
                                                          distance_matrix)
            travel_gar = datetime.timedelta(minutes=(travel_gar_to_loc))
            print(f"        - Travel time from garage to destination: {travel_gar_to_loc} mins")

            garage_leave_time = row_values["departure_time"] - travel_gar  # When bus must leave garage
            print(f"        - Bus must leave garage at: {garage_leave_time}")
            time_charging = garage_leave_time - most_recent_arrival_final  # Calculated the total time the bus would spend charging assuming it leaves early to be on departure on time
            time_charging_hours = time_charging.total_seconds() / 3600
            print(f"        - Available charging time: {time_charging_hours} hours ({time_charging})")

            charged_amount = time_charging_hours * 450
            print(f"        - Chargeable amount: {charged_amount}")
            bus_pre_depart_energy = min(charging_dict[bus_id][-1]["ze_energy"] + charged_amount,
                                        229.5)  # Use the last recorded energy level
            print(f"        - Energy after charging: {bus_pre_depart_energy}")

            if bus_pre_depart_energy >= 153:
                print(f"        - Bus {bus_id} is VIABLE (energy: {bus_pre_depart_energy} >= 153)")
                q = True
                viable_bus_ids[bus_id] = bus_pre_depart_energy
            else:
                print(f"        - Bus {bus_id} is NOT viable (energy: {bus_pre_depart_energy} < 153)")
                q = False

        if not q:
            # No viable buses: create new one
            print(f"      - No viable buses, creating new bus")
            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            print(f"      - New bus ID: {new_bus_id}")
            locations[row_values["start"]].append(new_bus_id)
            charging_record = None
            record, _ = create_material_record('ehvgar', row_values["start"], row_values["line"],
                                               row_values["departure_time"], distance_matrix, new_bus_id,
                                               255)  # new bus always start at max of SoH
        else:
            print(f"      - Viable buses found, selecting best")
            best_bus = max(viable_bus_ids, key=viable_bus_ids.get)
            print(f"      - Selected Bus {best_bus} with energy {viable_bus_ids[best_bus]}")
            charging_record = create_charging_record(most_recent_arrival_final, (time_charging_hours * 60), "999",
                                                     best_bus, bus_pre_depart_energy)
            locations["ehvgar"].remove(best_bus)
            locations[row_values["start"]].append(best_bus)
            print(f"      - Moved Bus {best_bus} to {row_values['start']}")

            bus_pre_depart_energy = viable_bus_ids[best_bus]
            record, _ = create_material_record('ehvgar', row_values["start"], row_values["line"],
                                               row_values["departure_time"], distance_matrix, best_bus,
                                               bus_pre_depart_energy)
        return record, charging_record

import numpy as np

def main_timetable_iteration(timetable, distance_matrix, locations, charging_dict):
    print("[MAIN ITERATION] Starting timetable iteration")
    generated_data = pd.DataFrame(
        columns=["start location", "end location", "start time", "end time", "activity", "line", "energy consumption",
                 "bus"])

    for index, row in timetable.iterrows():
        print(f"\n[ITERATION {index}] Processing timetable row {index}")
        timetable_cols = ["start", "departure_time", "end", "line"]
        row_vals = dict.fromkeys(timetable_cols)

        for column in timetable_cols:
            row_vals[column] = timetable.at[index, column]


        if not generated_data.empty:
            z = row_vals["start"]
            bussin = (locations[z])
            bus_id += 0

            current_energy = generated_data["energy consumption"].iloc[
                -1]  # Currently this does not handle multiple bus-lines in timetable or the first line not existing
            print(f"  - Using last bus: {bus_id}, energy: {current_energy}")
        else:
            bus_id = 1
            current_energy = 255
            print(f"  - First iteration, initializing bus {bus_id} with energy 255")


        print(
            f"  - Trip details: {row_vals['start']} -> {row_vals['end']}, Line {row_vals['line']}, Depart {row_vals['departure_time']}")

        busses_at_current_location = locations[row_vals["start"]]
        print(f"  - Buses at {row_vals['start']}: {busses_at_current_location}")

        if not busses_at_current_location:  # FOR SOME REASON ONLY EVER RUNS(CHARGES) TWICE??
            print(f"  - No buses at location, creating garage trips")
            pre_trip, charging_trip = create_garage_trips(timetable, distance_matrix, index, locations, charging_dict)

            print(f"  - charging_trip: {charging_trip}")
            print(f"  - pre_trip: {pre_trip}")

            trip = create_service_record(row_vals["start"], row_vals["end"], row_vals["departure_time"],
                                         row_vals["line"], bus_id, distance_matrix, current_energy)
            if charging_trip is not None:
                print(f"  - Adding charging, pre-trip, and service trip records")
                generated_data = pd.concat([
                    generated_data,
                    pd.DataFrame([charging_trip], columns=generated_data.columns),
                    pd.DataFrame([pre_trip], columns=generated_data.columns),
                    pd.DataFrame([trip], columns=generated_data.columns)
                ], ignore_index=True)
            else:
                print(f"  - Adding pre-trip and service trip records (no charging)")
                generated_data = pd.concat([
                    generated_data,
                    pd.DataFrame([pre_trip], columns=generated_data.columns),
                    pd.DataFrame([trip], columns=generated_data.columns)
                ], ignore_index=True)

        else:
            if current_energy > 51:  # 20% (at this point we just hope it doesn't dip below 10%)
                print(f"  - Bus has sufficient energy ({current_energy} > 51), proceeding with service trip")
                current_energy = generated_data["energy consumption"].iloc[-1]
                trip = create_service_record(row_vals["start"], row_vals["end"], row_vals["departure_time"],
                                             row_vals["line"], bus_id, distance_matrix, current_energy)
                generated_data = pd.concat([generated_data, pd.DataFrame([trip], columns=generated_data.columns)],
                                           ignore_index=True)
            else:
                print(f"  - Bus has low energy ({current_energy} <= 51), sending to garage")
                send_to_garage, _ = create_garage_trips(timetable, distance_matrix, index, locations, charging_dict,
                                                        bus_id, current_energy)
                pre_trip, _ = create_garage_trips(timetable, distance_matrix, index, locations, charging_dict)
                generated_data = pd.concat([
                    generated_data,
                    pd.DataFrame([send_to_garage], columns=generated_data.columns),
                    pd.DataFrame([pre_trip], columns=generated_data.columns)
                ], ignore_index=True)

    print("\n[MAIN ITERATION] Completed timetable iteration")
    return generated_data


def main(timetable, distance_matrix):
    print("[CREATE PLANNING] Initializing planning process")
    locations = {loc: [] for loc in set(timetable['start']).union(timetable['end'])}
    print(f"  - Initialized locations: {list(locations.keys())}")
    charging_dict = {}
    print(f"  - Initialized charging dictionary")

    print(f"  - Renaming and sorting timetable by departure_time")
    timetable = (rename_time_object(timetable, "departure_time", "Not Inside"))# .sort_values(by="departure_time")
    print(f"  - Timetable sorted, {len(timetable)} rows")

    print(f"  - Starting main timetable iteration")
    generated_data = main_timetable_iteration(timetable, distance_matrix, locations, charging_dict)

    print(f"  - Validating generated data structure")
    validate_dataframe_structure(generated_data, apply=True)
    generated_data['start time'] = pd.to_datetime(generated_data['start time'])
    generated_data['end time'] = pd.to_datetime(generated_data['end time'])

    print(f"  - Writing output to Excel: Excel Files/CREATED.xlsx")
    generated_data.to_excel("../Excel Files/CREATED.xlsx", index=False)
    print(f"[CREATE PLANNING] Planning complete! Generated {len(generated_data)} records")

    return generated_data

if __name__ == "__main__":
    PLANNING = pd.read_excel('../Excel Files/Bus Planning.xlsx')
    TIMETABLE = pd.read_excel('../Excel Files/Timetable.xlsx')
    DISTANCEMATRIX = pd.read_excel('../Excel Files/DistanceMatrix.xlsx')
    main(TIMETABLE, DISTANCEMATRIX)