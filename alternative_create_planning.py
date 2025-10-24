import pandas as pd

# --- Helper functions ---
def parse_time(t, service_start=5):
    """Convert HH:MM to minutes since start of service (05:00)."""
    h, m = map(int, t.split(":"))
    minutes = h * 60 + m
    # If before service_start, treat as next day
    if minutes < service_start * 60:
        minutes += 24 * 60
    return minutes

def format_minutes(minutes):
    """Convert minutes to HH:MM string, wrapping past 24h."""
    minutes %= 24 * 60
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

# --- Main planning function ---
def plan_buses(departures, travel_times, garage_location="ehvgar", garage_travel_times=None):
    buses = []
    schedule = []

    # Sort departures by time
    departures.sort(key=lambda d: parse_time(d["departure_time"]))

    for dep in departures:
        route = dep["line"]
        dep_time = parse_time(dep["departure_time"])
        from_loc = dep["start"]
        to_loc = dep["end"]

        travel_time = travel_times[route]
        arrival_time = dep_time + travel_time

        assigned_bus = None

        # Pick earliest available bus at the location
        available_buses = [
            (i, bus) for i, bus in enumerate(buses)
            if bus["location"] == from_loc and bus["available_at"] <= dep_time
        ]
        if available_buses:
            assigned_bus, bus = min(available_buses, key=lambda x: x[1]["available_at"])
            buses[assigned_bus]["available_at"] = arrival_time
            buses[assigned_bus]["location"] = to_loc
        else:
            assigned_bus = len(buses)

            # Add garage trip if defined
            if garage_travel_times and (garage_location, from_loc) in garage_travel_times:
                garage_time = garage_travel_times[(garage_location, from_loc)]
                garage_dep = dep_time - garage_time
                garage_arrival = dep_time

                schedule.append({
                    "route": "",
                    "start": garage_location,
                    "end": from_loc,
                    "departure": format_minutes(garage_dep),
                    "arrival": format_minutes(garage_arrival),
                    "bus": assigned_bus + 1
                })

            buses.append({
                "available_at": arrival_time,
                "location": to_loc
            })

        # Add main route trip
        schedule.append({
            "route": route,
            "start": from_loc,
            "end": to_loc,
            "departure": format_minutes(dep_time),
            "arrival": format_minutes(arrival_time),
            "bus": assigned_bus + 1
        })

    # Return trips to garage
    last_trip_index = {}
    for idx, trip in enumerate(schedule):
        last_trip_index[trip['bus']] = idx

    for bus_num, idx in last_trip_index.items():
        last_trip = schedule[idx]
        last_loc = last_trip['end']
        dep_time = parse_time(last_trip['arrival'])

        if garage_travel_times and (last_loc, garage_location) in garage_travel_times:
            return_time = dep_time + garage_travel_times[(last_loc, garage_location)]
            schedule.insert(idx + 1, {
                "route": "",
                "start": last_loc,
                "end": garage_location,
                "departure": format_minutes(dep_time),
                "arrival": format_minutes(return_time),
                "bus": bus_num
            })
            for k in last_trip_index:
                if last_trip_index[k] > idx:
                    last_trip_index[k] += 1

    return schedule, len(buses)

# --- Wrapper to run scheduler and save to Excel ---
def run_bus_scheduler(departures_file, travel_times_file, output_file, garage_location="ehvgar"):
    df_departures = pd.read_excel(departures_file)
    required_cols = {'line', 'start', 'end', 'departure_time'}
    if not required_cols.issubset(df_departures.columns):
        raise ValueError(f"Timetable file must include columns: {required_cols}")

    departures = df_departures.to_dict(orient='records')

    # Travel times

    df_travel = pd.read_excel(travel_times_file)
    df_travel = df_travel.dropna(subset=['start', 'end', 'max_travel_time'])

    # Line travel times (in minutes)
    travel_times = {
        row['line']: row['max_travel_time'] for _, row in df_travel.iterrows() if pd.notna(row['line'])
    }

    # Garage travel times (line is NaN)
    garage_travel_times = {
        (row['start'], row['end']): row['max_travel_time']
        for _, row in df_travel.iterrows() if pd.isna(row['line'])
    }

    # Run scheduler
    schedule, total_buses = plan_buses(departures, travel_times, garage_location, garage_travel_times)

    # Convert to DataFrame
    df_schedule = pd.DataFrame(schedule)

    # Add activity column
    df_schedule['activity'] = df_schedule.apply(
        lambda row: 'material trip' if row['start'] == garage_location or row['end'] == garage_location else 'service trip',
        axis=1
    )

    # Add blank energy consumption column
    df_schedule['energy consumption'] = "1"

    # Rename columns
    df_schedule = df_schedule.rename(columns={
        "route": "line",
        "start": "start location",
        "end": "end location",
        "departure": "start time",
        "arrival": "end time"
    })

  # --- Sort by bus and actual minutes, not formatted string ---
    df_schedule['start_minutes'] = [parse_time(t) for t in df_schedule['start time']]
    df_schedule = df_schedule.sort_values(by=['bus', 'start_minutes']).reset_index(drop=True)
    df_schedule = df_schedule.drop(columns=['start_minutes'])
    # Save to Excel
    df_schedule.to_excel(output_file, index=False)

    print(f"✅ Schedule written to '{output_file}'")
    print(f"🚌 Total buses used: {total_buses}")

run_bus_scheduler(
    departures_file=r"Excel Files/Timetable.xlsx",
    travel_times_file=r"Excel Files/DistanceMatrix.xlsx",
    output_file="Excel Files/floodbusseswithout_battery.xlsx"
)