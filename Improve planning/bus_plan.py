#!/usr/bin/env python3
"""
run_bus_planner_charge_show.py

Behavior:
 - Reads assignment/Timetable.xlsx and assignment/DistanceMatrix.xlsx and Bus Planning.xlsx (if present).
 - Schedules trips, inserts material/idle/charging events.
 - If a bus cannot start a trip because SOC is too low, that bus will be sent to the charging location ("edhgar")
   (deadhead -> charging) and will NOT be used for that trip. A NEW bus (starting at edhgar fully charged) will be
   dispatched to perform the trip (material -> service). This increases bus count when charging is required.
 - Outputs Excel file with columns:
     start location | end location | start time | end time | activity | line | energy consumption | bus
   (all columns have values; "energy consumption" left empty "")
 - Outputs interactive gantt html showing service/material/idle/charging.

Usage:
  python run_bus_planner_charge_show.py
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import math

# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = "assignment"
OUTPUT_DIR = "output"
TIMETABLE_FILE = "Timetable.xlsx"
DIST_FILE = "DistanceMatrix.xlsx"
BUSPLAN_FILE = "Bus Planning.xlsx"  # optional

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Energy / SOC parameters
BATTERY_CAPACITY_KWH = 300.0        # kWh
CONSUMPTION_KWH_PER_KM = 1.5        # kWh/km (used for service & material)
IDLE_KWH_PER_MIN = 0.02             # kWh per minute idle
CHARGE_KWH_PER_MIN = 3.0            # kWh gained per minute of charging (~180 kW)
SOC_MIN = 0.10                       # 10%
SOC_THRESHOLD = 0.25                 # if SOC below this, bus should go charge (and not be used)
SOC_TARGET_AFTER_CHARGE = 0.90       # charge up to this SOC when sending to charge
SOC_FULL = 1.0

AVERAGE_SPEED_KMPH = 30.0
DEFAULT_TRIP_MINUTES = 10
DEADHEAD_DEFAULT_MINUTES = 5

CHARGING_LOCATION = "edhgar"  # per your message

VERBOSE = True

def log(*args):
    if VERBOSE:
        print(*args)

# ----------------------------
# Utilities
# ----------------------------
def parse_time_like(x, base_date):
    if pd.isna(x):
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        dt = x.to_pydatetime() if isinstance(x, pd.Timestamp) else x
        return datetime.combine(base_date.date(), dt.time())
    try:
        dt = pd.to_datetime(str(x), errors='coerce')
        if pd.isna(dt):
            return None
        return datetime.combine(base_date.date(), dt.to_pydatetime().time())
    except:
        return None

def safe_str(x):
    return "" if pd.isna(x) else str(x)

# ----------------------------
# Load files
# ----------------------------
timetable_path = os.path.join(INPUT_DIR, TIMETABLE_FILE)
dist_path = os.path.join(INPUT_DIR, DIST_FILE)
busplan_path = os.path.join(INPUT_DIR, BUSPLAN_FILE)

if not os.path.exists(timetable_path):
    raise FileNotFoundError(f"Timetable file not found at {timetable_path}")
if not os.path.exists(dist_path):
    raise FileNotFoundError(f"Distance matrix file not found at {dist_path}")

timetable_sheets = pd.read_excel(timetable_path, sheet_name=None)
timetable_df = None
# choose plausible sheet
for name, df in timetable_sheets.items():
    low = name.lower()
    if "timetable" in low or "time" in low or "sheet1" in low:
        timetable_df = df
        break
if timetable_df is None:
    timetable_df = list(timetable_sheets.values())[0]

dist_sheets = pd.read_excel(dist_path, sheet_name=None)
dist_df = None
for name, df in dist_sheets.items():
    low = name.lower()
    if "distance" in low or "matrix" in low or "dist" in low:
        dist_df = df
        break
if dist_df is None:
    dist_df = list(dist_sheets.values())[0]

# optional busplan (for matching activities) - not strictly required
busplan_df = None
if os.path.exists(busplan_path):
    try:
        bps = pd.read_excel(busplan_path, sheet_name=None)
        busplan_df = list(bps.values())[0]
    except Exception:
        busplan_df = None

# ----------------------------
# Parse timetable into trip list
# ----------------------------
base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# heuristics for columns
cols = [c.lower() for c in timetable_df.columns]
def find_col(keywords):
    for k in keywords:
        for i,c in enumerate(cols):
            if k in c:
                return timetable_df.columns[i]
    return None

col_origin = find_col(["start location","origin","from","start"])
col_dest   = find_col(["end location","destination","dest","to","end"])
col_start  = find_col(["start time","departure","dep","time"])
col_end    = find_col(["end time","arrival","arr","end"])
col_line   = find_col(["line","route"])

# fallback column picks
cols_all = list(timetable_df.columns)
if col_origin is None:
    col_origin = cols_all[0]
if col_dest is None and len(cols_all) >= 3:
    col_dest = cols_all[2]
if col_start is None and len(cols_all) >= 2:
    col_start = cols_all[1]
if col_line is None:
    col_line = col_line  # may remain None

trips = []
for idx, row in timetable_df.iterrows():
    start_t = parse_time_like(row.get(col_start), base_date)
    end_t = parse_time_like(row.get(col_end), base_date) if col_end is not None else None
    if start_t is None:
        continue
    if end_t is None:
        end_t = start_t + timedelta(minutes=DEFAULT_TRIP_MINUTES)
    trip = {
        "trip_id": idx,
        "start location": safe_str(row.get(col_origin)),
        "end location": safe_str(row.get(col_dest)),
        "start time": start_t,
        "end time": end_t,
        "line": safe_str(row.get(col_line)) if col_line is not None else "Line A"
    }
    trips.append(trip)

trips = sorted(trips, key=lambda x: x["start time"])
log("Parsed trips:", len(trips))

# ----------------------------
# Parse distance matrix into lookup (assume first column could be index)
# ----------------------------
df = dist_df.copy()
# If first column looks like labels, set index
if df.shape[1] > 1 and df.iloc[:,0].dtype == object:
    try:
        df = df.set_index(df.columns[0])
    except Exception:
        pass
df.index = df.index.astype(str)
df.columns = df.columns.astype(str)
dist_numeric = df.apply(pd.to_numeric, errors='coerce')

def get_distance_km(a,b):
    try:
        if a is None or b is None:
            return None
        a_str = str(a); b_str = str(b)
        v = dist_numeric.loc[a_str, b_str]
        if pd.isna(v): return None
        return float(v)
    except Exception:
        return None

def distance_to_minutes(km):
    if km is None or np.isnan(km):
        return DEADHEAD_DEFAULT_MINUTES
    return km / AVERAGE_SPEED_KMPH * 60.0

def estimate_deadhead_minutes(a,b):
    km = get_distance_km(a,b)
    if km is None:
        return DEADHEAD_DEFAULT_MINUTES
    return distance_to_minutes(km)

def energy_needed_kwh(a,b):
    km = get_distance_km(a,b)
    if km is None:
        # conservative 1 km -> consumption
        km = 1.0
    return km * CONSUMPTION_KWH_PER_KM

# ----------------------------
# Scheduler state & helpers
# ----------------------------
vehicles = []  # list of dicts: id, soc (0-1), available(datetime), location, schedule(list of events)
next_bus_id = 1

def create_new_bus(at_location, available_time):
    global next_bus_id
    v = {
        "id": next_bus_id,
        "soc": SOC_FULL,  # starts fully charged
        "available": available_time,
        "location": at_location,
        "schedule": []
    }
    next_bus_id += 1
    vehicles.append(v)
    return v

def add_event(v, start_loc, end_loc, start_t, end_t, activity, line, bus_label=None, soc_end_pct=None):
    # energy consumption left empty as requested
    ev = {
        "start location": start_loc,
        "end location": end_loc,
        "start time": start_t,
        "end time": end_t,
        "activity": activity,
        "line": line if line is not None else "Line A",
        "energy consumption": "",   # left empty intentionally
        "bus": bus_label if bus_label is not None else f"Bus {v['id']}",
        "SOC end (%)": round(soc_end_pct*100,1) if soc_end_pct is not None else round(v["soc"]*100,1)
    }
    v["schedule"].append(ev)

def charge_minutes_needed(current_soc, target_soc=SOC_TARGET_AFTER_CHARGE):
    if current_soc >= target_soc:
        return 0
    needed_kwh = (target_soc - current_soc) * BATTERY_CAPACITY_KWH
    minutes = math.ceil(needed_kwh / CHARGE_KWH_PER_MIN)
    return minutes

# ----------------------------
# Main greedy scheduling with "send bus to charge and use new bus" policy
# ----------------------------
for trip in trips:
    assigned = False
    # try to find an existing bus that can do the trip without needing charging beforehand
    for v in vehicles:
        # compute deadhead minutes and earliest arrival if leaving immediately
        dead_min = estimate_deadhead_minutes(v["location"], trip["start location"])
        arrival_if_leave = v["available"] + timedelta(minutes=dead_min)
        if arrival_if_leave > trip["start time"]:
            continue  # can't make it in time

        # energy needed: deadhead + trip
        energy_dead = energy_needed_kwh(v["location"], trip["start location"])
        energy_trip = energy_needed_kwh(trip["start location"], trip["end location"])
        total_energy = energy_dead + energy_trip

        # SOC in kWh:
        soc_kwh = v["soc"] * BATTERY_CAPACITY_KWH

        if soc_kwh >= total_energy and v["soc"] >= SOC_THRESHOLD:
            # assign this bus:
            # 1) material deadhead if needed
            if v["location"] != trip["start location"]:
                dh_start = v["available"]
                dh_end = dh_start + timedelta(minutes=dead_min)
                # update soc
                v["soc"] -= energy_dead / BATTERY_CAPACITY_KWH
                add_event(v, v["location"], trip["start location"], dh_start, dh_end, "material", trip["line"], f"Bus {v['id']}", v["soc"])
                v["available"] = dh_end
                v["location"] = trip["start location"]

            # 2) idle if early
            if v["available"] < trip["start time"]:
                idle_start = v["available"]
                idle_end = trip["start time"]
                idle_min = (idle_end - idle_start).total_seconds()/60.0
                # apply idle consumption
                v["soc"] -= (idle_min * IDLE_KWH_PER_MIN) / BATTERY_CAPACITY_KWH
                add_event(v, trip["start location"], trip["start location"], idle_start, idle_end, "idle", trip["line"], f"Bus {v['id']}", v["soc"])
                v["available"] = idle_end

            # 3) service trip
            v["soc"] -= energy_trip / BATTERY_CAPACITY_KWH
            add_event(v, trip["start location"], trip["end location"], trip["start time"], trip["end time"], "service", trip["line"], f"Bus {v['id']}", v["soc"])
            v["available"] = trip["end time"]
            v["location"] = trip["end location"]
            assigned = True
            break
        else:
            # Bus cannot perform the trip without charging first. According to your rule:
            # send this bus to charging location (deadhead -> charge) and DO NOT use it for this trip.
            # We create events: deadhead to charging location (material), charging event; update its state.
            # Then we will create a NEW bus at charging location to service the trip.
            # 1) deadhead from current location to charging location
            dh_to_charge_min = estimate_deadhead_minutes(v["location"], CHARGING_LOCATION)
            dh_start = v["available"]
            dh_end = dh_start + timedelta(minutes=dh_to_charge_min)
            energy_dh_to_charge = energy_needed_kwh(v["location"], CHARGING_LOCATION)
            v["soc"] -= energy_dh_to_charge / BATTERY_CAPACITY_KWH
            add_event(v, v["location"], CHARGING_LOCATION, dh_start, dh_end, "material", trip["line"], f"Bus {v['id']}", v["soc"])
            # 2) charging minutes needed to reach target
            minutes_to_charge = charge_minutes_needed(v["soc"], SOC_TARGET_AFTER_CHARGE)
            if minutes_to_charge <= 0:
                minutes_to_charge = 15  # minimum session
            charge_start = dh_end
            charge_end = charge_start + timedelta(minutes=minutes_to_charge)
            energy_gain_kwh = minutes_to_charge * CHARGE_KWH_PER_MIN
            v["soc"] = min(SOC_FULL, v["soc"] + energy_gain_kwh / BATTERY_CAPACITY_KWH)
            add_event(v, CHARGING_LOCATION, CHARGING_LOCATION, charge_start, charge_end, "charging", trip["line"], f"Bus {v['id']}", v["soc"])
            v["available"] = charge_end
            v["location"] = CHARGING_LOCATION
            # Now this bus is unavailable for the trip; continue loop to try other buses (but ensure we don't try same bus again for this trip)
            # NOTE: do not assign this v to the trip. We'll assign a new bus below.
            # continue searching other buses
            continue

    if not assigned:
        # create new bus at charging location (as specified) with full SOC and send it to do the trip.
        # new bus starts at CHARGING_LOCATION, depart earlier to reach trip start
        new_bus = create_new = None
        # create bus at charging location with available time sufficiently early (we'll set available to trip start - deadhead - small buffer)
        # compute deadhead minutes from charging location to trip start
        dh_from_charge_min = estimate_deadhead_minutes(CHARGING_LOCATION, trip["start location"])
        # set new bus available time so that if it leaves charging location at available -> arrives at trip.start
        new_available = trip["start time"] - timedelta(minutes=dh_from_charge_min)
        # create bus
        new_bus = {
            "id": next_bus_id,
            "soc": SOC_FULL,
            "available": new_available,
            "location": CHARGING_LOCATION,
            "schedule": []
        }
        # increment id
        next_bus_id += 1
        vehicles.append(new_bus)

        # material: charging_location -> trip start
        dh_start = new_bus["available"]
        dh_end = dh_start + timedelta(minutes=dh_from_charge_min)
        energy_dh = energy_needed_kwh(CHARGING_LOCATION, trip["start location"])
        new_bus["soc"] -= energy_dh / BATTERY_CAPACITY_KWH
        add_event(new_bus, CHARGING_LOCATION, trip["start location"], dh_start, dh_end, "material", trip["line"], f"Bus {new_bus['id']}", new_bus["soc"])
        new_bus["available"] = dh_end
        new_bus["location"] = trip["start location"]

        # idle if early
        if new_bus["available"] < trip["start time"]:
            idle_start = new_bus["available"]
            idle_end = trip["start time"]
            idle_min = (idle_end - idle_start).total_seconds()/60.0
            new_bus["soc"] -= (idle_min * IDLE_KWH_PER_MIN) / BATTERY_CAPACITY_KWH
            add_event(new_bus, trip["start location"], trip["start location"], idle_start, idle_end, "idle", trip["line"], f"Bus {new_bus['id']}", new_bus["soc"])
            new_bus["available"] = idle_end

        # service
        energy_trip = energy_needed_kwh(trip["start location"], trip["end location"])
        new_bus["soc"] -= energy_trip / BATTERY_CAPACITY_KWH
        add_event(new_bus, trip["start location"], trip["end location"], trip["start time"], trip["end time"], "service", trip["line"], f"Bus {new_bus['id']}", new_bus["soc"])
        new_bus["available"] = trip["end time"]
        new_bus["location"] = trip["end location"]
        assigned = True

# After scheduling all trips, ensure each bus has at least one idle and charging at end to return to depot if needed
for v in vehicles:
    # add a short idle
    idle_start = v["available"]
    idle_end = idle_start + timedelta(minutes=10)
    # idle energy
    idle_minutes = (idle_end - idle_start).total_seconds()/60.0
    v["soc"] -= (idle_minutes * IDLE_KWH_PER_MIN) / BATTERY_CAPACITY_KWH
    add_event(v, v["location"], v["location"], idle_start, idle_end, "idle", "End", f"Bus {v['id']}", v["soc"])
    v["available"] = idle_end
    # if SOC < 0.8, send to charge (material->charging)
    if v["soc"] < 0.8:
        # deadhead to charging location if not already there
        if v["location"] != CHARGING_LOCATION:
            dh_min = estimate_deadhead_minutes(v["location"], CHARGING_LOCATION)
            dh_start = v["available"]
            dh_end = dh_start + timedelta(minutes=dh_min)
            energy_dh = energy_needed_kwh(v["location"], CHARGING_LOCATION)
            v["soc"] -= energy_dh / BATTERY_CAPACITY_KWH
            add_event(v, v["location"], CHARGING_LOCATION, dh_start, dh_end, "material", "End", f"Bus {v['id']}", v["soc"])
            v["available"] = dh_end
            v["location"] = CHARGING_LOCATION
        # charging
        minutes_to_full = math.ceil(((SOC_FULL - v["soc"]) * BATTERY_CAPACITY_KWH) / CHARGE_KWH_PER_MIN)
        if minutes_to_full <= 0:
            minutes_to_full = 15
        start_c = v["available"]
        end_c = start_c + timedelta(minutes=minutes_to_full)
        energy_gain = minutes_to_full * CHARGE_KWH_PER_MIN
        v["soc"] = min(SOC_FULL, v["soc"] + energy_gain / BATTERY_CAPACITY_KWH)
        add_event(v, CHARGING_LOCATION, CHARGING_LOCATION, start_c, end_c, "charging", "End", f"Bus {v['id']}", v["soc"])
        v["available"] = end_c
        v["location"] = CHARGING_LOCATION

# ----------------------------
# Collect all events and build final DataFrame
# ----------------------------
all_rows = []
for v in vehicles:
    for ev in v["schedule"]:
        # Ensure required columns present and not null
        row = {
            "start location": ev.get("start location", "") or "",
            "end location": ev.get("end location", "") or "",
            "start time": ev.get("start time"),
            "end time": ev.get("end time"),
            "activity": ev.get("activity", "") or "",
            "line": ev.get("line", "") or "",
            "energy consumption": ev.get("energy consumption", "") if ev.get("energy consumption") is not None else "",
            "bus": ev.get("bus", "") or "",
            # keep SOC for inspection but not required
            "SOC end (%)": ev.get("SOC end (%)", "")
        }
        # Fill missing textual items
        for k in ["start location", "end location", "activity", "line", "energy consumption", "bus"]:
            if row[k] is None or (isinstance(row[k], float) and np.isnan(row[k])):
                row[k] = ""
        # ensure times are datetimes
        if not isinstance(row["start time"], datetime):
            row["start time"] = pd.to_datetime(row["start time"])
        if not isinstance(row["end time"], datetime):
            row["end time"] = pd.to_datetime(row["end time"])
        all_rows.append(row)

df_out = pd.DataFrame(all_rows)

# Required final column order:
final_cols = ["start location", "end location", "start time", "end time", "activity", "line", "energy consumption", "bus"]
# Ensure all columns exist, fill blanks for any missing
for c in final_cols:
    if c not in df_out.columns:
        df_out[c] = ""
# Reorder and keep SOC end (%) at the end for inspection
out_df = df_out[final_cols + ["SOC end (%)"]]

# Replace any NaNs with empty strings (energy consumption empty)
out_df = out_df.fillna("")
# Ensure every required column has a non-empty value; replace empty line with "Line A", empty bus with "Unassigned"
out_df["start location"] = out_df["start location"].replace("", "unknown")
out_df["end location"] = out_df["end location"].replace("", "unknown")
out_df["activity"] = out_df["activity"].replace("", "idle")
out_df["line"] = out_df["line"].replace("", "Line A")
out_df["energy consumption"] = out_df["energy consumption"].replace("", "")
out_df["bus"] = out_df["bus"].replace("", "Unassigned")

# ----------------------------
# Save Excel
# ----------------------------
excel_path = os.path.join(OUTPUT_DIR, "improved_bus_schedule_charge_show.xlsx")
# Convert datetimes to Excel-friendly format if necessary
out_df.to_excel(excel_path, index=False)
log("Saved schedule to", excel_path)

# ----------------------------
# Gantt chart showing all 4 activities (service/material/idle/charging)
# ----------------------------
gantt = out_df.copy()
gantt["start_dt"] = pd.to_datetime(gantt["start time"])
gantt["end_dt"] = pd.to_datetime(gantt["end time"])
# normalize activity labels
gantt["activity_norm"] = gantt["activity"].str.lower().replace({
    "service":"service",
    "material":"material",
    "idle":"idle",
    "charging":"charging"
}).fillna("idle")

color_map = {
    "service": "green",
    "material": "orange",
    "idle": "gray",
    "charging": "blue"
}

fig = px.timeline(
    gantt,
    x_start="start_dt",
    x_end="end_dt",
    y="bus",
    color="activity_norm",
    color_discrete_map=color_map,
    hover_data=["start location", "end location", "line", "SOC end (%)"],
    title="Bus schedule with service/material/idle/charging"
)

fig.update_yaxes(title="Bus", autorange="reversed")
fig.update_layout(height=800)
gantt_html = os.path.join(OUTPUT_DIR, "gantt_charge_show.html")
fig.write_html(gantt_html, include_plotlyjs="cdn")
log("Saved gantt to", gantt_html)

log("Done. Buses used:", len(vehicles))
