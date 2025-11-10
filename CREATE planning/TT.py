import pandas as pd
import datetime


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

        locations.setdefault("ehvgar", []).append(bus_id)  # Add to garage
        print(f"      - Added Bus {bus_id} to garage")

        # Create a charging record and log the trip to the garage
        record, cur_energy = create_material_record(row_values["start"], 'ehvgar', row_values["line"],
                                                    row_values["departure_time"], distance_matrix, bus_id,
                                                    current_energy)

        # Record bus arrival at garage
        if row_values["end"] == "ehvgar":
            print(f"      - Trip ends at garage, recording arrival")
            record_garage_arrival(charging_dict, bus_id, record[3], cur_energy)
            if bus_id in locations[row_values["start"]]:
                locations[row_values["start"]].remove(bus_id)
            locations.setdefault("ehvgar", []).append(bus_id)

        return record, None

    else:
        print(f"    - No bus ID provided, checking garage status")

        # Check if there are any buses in the garage
        if not locations.get("ehvgar", []):  # Garage is empty
            print(f"      - Garage empty, creating new bus")
            if row_values["start"] in locations and bus_id in locations[row_values["start"]]:
                locations[row_values["start"]].remove(bus_id)

            # Create new bus
            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            print(f"      - New bus ID: {new_bus_id}")
            locations[row_values["start"]].append(new_bus_id)

            max_tt, _ = lookup_distance_matrix('ehvgar', row_values["start"], row_values["line"], distance_matrix)
            pre_trip_depart = row_values["departure_time"] - datetime.timedelta(
                minutes=max_tt)  # Adjust for travel time
            record, _ = create_material_record('ehvgar', row_values["start"], row_values["line"], pre_trip_depart,
                                               distance_matrix, new_bus_id, 255)  # new bus starts at full charge
            return record, None

        # If garage has buses: evaluate viability
        print(f"    - Garage has buses, evaluating viability")
        garage_buses = locations["ehvgar"]
        print(f"      - Buses in garage: {garage_buses}")
        viable_bus_ids = {}

        for bus_id in garage_buses:
            print(f"      - Evaluating Bus {bus_id}")
            # Get charging records for the bus
            bus_charging_records = charging_dict.get(bus_id, [])
            if not bus_charging_records:
                print(f"        - No charging records for Bus {bus_id}, skipping")
                continue

            most_recent_arrival_str = max(item["arrival_time"] for item in charging_dict[bus_id])
            most_recent_arrival_dt = datetime.datetime.strptime(most_recent_arrival_str, "%H:%M:%S")
            most_recent_arrival_final = pd.Timestamp(most_recent_arrival_dt)

            travel_gar_to_loc, _ = lookup_distance_matrix('ehvgar', row_values["end"], row_values["line"],
                                                          distance_matrix)
            travel_gar = datetime.timedelta(minutes=travel_gar_to_loc)
            garage_leave_time = row_values["departure_time"] - travel_gar
            time_charging = garage_leave_time - most_recent_arrival_final
            time_charging_hours = time_charging.total_seconds() / 3600

            print(f"        - Available charging time: {time_charging_hours} hours")
            charged_amount = time_charging_hours * 450  # Charge per hour
            bus_pre_depart_energy = min(charging_dict[bus_id][-1]["ze_energy"] + charged_amount, 229.5)
            print(f"        - Energy after charging: {bus_pre_depart_energy}")

            if bus_pre_depart_energy >= 153:  # Energy threshold
                viable_bus_ids[bus_id] = bus_pre_depart_energy

        if viable_bus_ids:
            print(f"      - Viable buses found, selecting best")
            best_bus = max(viable_bus_ids, key=viable_bus_ids.get)
            print(f"      - Selected Bus {best_bus} with energy {viable_bus_ids[best_bus]}")
            charging_record = create_charging_record(most_recent_arrival_final, (time_charging_hours * 60), "999",
                                                     best_bus, bus_pre_depart_energy)
            locations["ehvgar"].remove(best_bus)
            locations[row_values["start"]].append(best_bus)

            # Create material record for best bus
            bus_pre_depart_energy = viable_bus_ids[best_bus]
            record, _ = create_material_record('ehvgar', row_values["start"], row_values["line"],
                                               row_values["departure_time"], distance_matrix, best_bus,
                                               bus_pre_depart_energy)
        else:
            # No viable buses, create new one
            print(f"      - No viable buses, creating new bus")
            all_bus_ids = [bid for loc_list in locations.values() for bid in loc_list]
            new_bus_id = max(all_bus_ids) + 1 if all_bus_ids else 1
            print(f"      - New bus ID: {new_bus_id}")
            locations[row_values["start"]].append(new_bus_id)
            charging_record = None
            record, _ = create_material_record('ehvgar', row_values["start"], row_values["line"],
                                               row_values["departure_time"], distance_matrix, new_bus_id, 255)

        return record, charging_record
