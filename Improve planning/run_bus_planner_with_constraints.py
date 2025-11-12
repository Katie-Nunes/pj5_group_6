#!/usr/bin/env python3
"""
run_bus_planner_with_activities.py

Reads:
  assignment/Timetable.xlsx
  assignment/DistanceMatrix.xlsx
  assignment/Bus Planning.xlsx

Produces:
  output/improved_bus_schedule_with_activities.xlsx
    - sheets: AssignedTrips (service only), AllTrips (service/material/idle/charging), VehicleSummary, OriginalTimetable, DistanceMatrix, BusPlanningSample
  output/gantt_with_activities.html  (interactive Plotly timeline visualizing service, material, idle (charging flagged) )

Behavior:
  - Uses greedy chaining with SOC/charging logic (same constraints as previous version).
  - PRESERVES activity labels from Bus Planning.xlsx where a timetable trip can be matched.
  - Explicitly inserts "material" trips for deadheads between trips and "idle" rows for waiting/charging.
  - Charging sessions are represented as idle rows with subactivity "charging" and a note.
  - Outputs an interactive Gantt colored by activity.

Usage:
  - Place this file next to your `assignment/` folder (same layout as before).
  - pip install pandas openpyxl plotly numpy
  - python run_bus_planner_with_activities.py
"""

import os
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    "INPUT_DIR": "../Excel Files",
    "TIMETABLE_FILE": "Timetable.xlsx",
    "DIST_FILE": "DistanceMatrix.xlsx",
    "BUSPLAN_FILE": "Bus Planning.xlsx",
    "OUTPUT_DIR": "../Excel Files",

    # battery & energy model
    "battery_capacity_kwh": 300.0,
    "initial_soc_percent": 100.0,
    "soh_percent": 85.0,
    "min_soc_percent_of_soh": 10.0,

    # consumption
    "consumption_kwh_per_km_default": 1.5,
    "idle_power_kw": 5.0,

    # charging
    "charging_location_hints": ["depot","garage","charging","charge","apt","ehvapt"],
    "charging_location": 'ehvgar',
    "charging_station_capacity": 40,
    "high_power_kw": 450.0,
    "low_power_kw": 60.0,
    "min_charging_minutes": 15,
    "charging_stop_prefers_90pct": True,

    # operation
    "default_trip_minutes_if_no_end": 10,
    "deadhead_default_minutes": 5,
    "speed_kmph_for_distance_to_time": 30.0,
    "consumption_variation_factor": 1.0,
    "verbose": True
}

# ----------------------------
# Utilities & IO
# ----------------------------
def log(*args, **kwargs):
    if CONFIG["verbose"]:
        print(*args, **kwargs)

INPUT_DIR = CONFIG["INPUT_DIR"]
OUT_DIR = CONFIG["OUTPUT_DIR"]
os.makedirs(OUT_DIR, exist_ok=True)

TIMETABLE_PATH = os.path.join(INPUT_DIR, CONFIG["TIMETABLE_FILE"])
DIST_PATH = os.path.join(INPUT_DIR, CONFIG["DIST_FILE"])
BUSPLAN_PATH = os.path.join(INPUT_DIR, CONFIG["BUSPLAN_FILE"])

def try_read_excel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_excel(path, sheet_name=None)

def find_best_sheet(sheet_dict, keywords):
    keys = list(sheet_dict.keys())
    key_l = [k.lower() for k in keys]
    for kw in keywords:
        for i,k in enumerate(key_l):
            if kw in k:
                return keys[i]
    return keys[0]

def parse_time(x):
    if pd.isna(x):
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.to_pydatetime()
    try:
        dt = pd.to_datetime(str(x), errors='coerce')
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except:
        return None

def combine_time_with_base(dt, base_date):
    if dt is None:
        return None
    if dt.date() != base_date.date():
        return datetime.combine(base_date.date(), dt.time())
    return dt

# ----------------------------
# Load files & detect sheets
# ----------------------------
log("Loading inputs...")
timetable_sheets = try_read_excel(TIMETABLE_PATH)
dist_sheets = try_read_excel(DIST_PATH)
busplan_sheets = try_read_excel(BUSPLAN_PATH)

timetable_sheet = find_best_sheet(timetable_sheets, ["timetable","schedule","times"])
dist_sheet = find_best_sheet(dist_sheets, ["distance","matrix","dist"])
busplan_sheet = find_best_sheet(busplan_sheets, ["bus","planning","vehicle","plan"])

timetable_df = timetable_sheets[timetable_sheet].copy()
dist_df = dist_sheets[dist_sheet].copy()
busplan_df = busplan_sheets[busplan_sheet].copy()

log("Using sheets:", timetable_sheet, dist_sheet, busplan_sheet)
log("Timetable columns:", list(timetable_df.columns))

# ----------------------------
# Column mapping heuristics
# ----------------------------
def find_col_like(df, hints):
    for h in hints:
        for c in df.columns:
            if h in str(c).lower():
                return c
    return None

col_origin = find_col_like(timetable_df, ["origin","start location","from","start"])
col_start = find_col_like(timetable_df, ["start time","departure","dep","time"])
col_dest = find_col_like(timetable_df, ["dest","destination","end","to"])
col_end = find_col_like(timetable_df, ["end time","arrival","arr","end"])

cols = list(timetable_df.columns)
if col_origin is None and len(cols) >= 1:
    col_origin = cols[0]
if col_start is None and len(cols) >= 2:
    col_start = cols[1]
if col_dest is None and len(cols) >= 3:
    col_dest = cols[2]

log("Mapped timetable cols -> origin:", col_origin, "start:", col_start, "dest:", col_dest, "end:", col_end)

# ----------------------------
# Build trips from timetable
# ----------------------------
BASE_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
trips = []
for idx,row in timetable_df.iterrows():
    raw_start = row.get(col_start) if col_start in row.index else None
    raw_origin = row.get(col_origin) if col_origin in row.index else None
    raw_dest = row.get(col_dest) if col_dest in row.index else None
    raw_end = row.get(col_end) if (col_end and col_end in row.index) else None

    start_dt = parse_time(raw_start)
    end_dt = parse_time(raw_end) if raw_end is not None else None
    if start_dt is not None:
        start_dt = combine_time_with_base(start_dt, BASE_DATE)
    if end_dt is not None:
        end_dt = combine_time_with_base(end_dt, BASE_DATE)
    if end_dt is None and start_dt is not None:
        end_dt = start_dt + timedelta(minutes=CONFIG["default_trip_minutes_if_no_end"])

    if start_dt is None:
        continue

    trips.append({
        "trip_row": idx,
        "origin": str(raw_origin) if pd.notna(raw_origin) else None,
        "start": start_dt,
        "dest": str(raw_dest) if pd.notna(raw_dest) else None,
        "end": end_dt,
        "raw": row.to_dict()
    })

trips = sorted(trips, key=lambda x: x["start"])
log("Parsed trips:", len(trips))

# ----------------------------
# Parse distance matrix
# ----------------------------
df = dist_df.copy()
if (df.shape[1] > 1) and (df.iloc[:,0].dtype == object):
    try:
        df = df.set_index(df.columns[0])
    except Exception:
        pass
df.index = df.index.astype(str)
df.columns = df.columns.astype(str)
dist_numeric = df.apply(pd.to_numeric, errors='coerce')

def get_distance_km(a,b):
    if a is None or b is None:
        return None
    a = str(a); b = str(b)
    try:
        v = dist_numeric.loc[a,b]
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return None

def distance_to_minutes_km(km):
    if km is None or np.isnan(km):
        return CONFIG["deadhead_default_minutes"]
    return km / CONFIG["speed_kmph_for_distance_to_time"] * 60.0

def estimate_deadhead_minutes(a,b):
    d = get_distance_km(a,b)
    if d is None:
        return CONFIG["deadhead_default_minutes"]
    return distance_to_minutes_km(d)

# ----------------------------
# Detect charging/depot location
# ----------------------------
if CONFIG["charging_location"] is None:
    detected = None
    candidates = list(dist_numeric.columns) + list(dist_numeric.index)
    cand_low = [c.lower() for c in candidates]
    for hint in CONFIG["charging_location_hints"]:
        for i,c in enumerate(cand_low):
            if hint in c:
                detected = candidates[i]
                break
        if detected:
            break
    if detected is None and len(candidates) > 0:
        detected = candidates[0]
    CONFIG["charging_location"] = detected

log("Charging/depot location:", CONFIG["charging_location"])

# ----------------------------
# Bus Planning activity mapping
# ----------------------------
# We'll try to map timetable rows to BusPlanning rows by matching start time, origin, dest (best-effort).
# Build an index from BusPlanning to activity by (start, origin, dest) keys normalized.
bp = busplan_df.copy()
# Normalize candidate fields (best-effort)
bp_cols = [c.lower() for c in bp.columns]
# heuristics for names
bp_origin_col = None
bp_start_col = None
bp_dest_col = None
bp_activity_col = None
for i,c in enumerate(bp.columns):
    cl = str(c).lower()
    if bp_origin_col is None and any(x in cl for x in ["start location","origin","from","start"]):
        bp_origin_col = c
    if bp_dest_col is None and any(x in cl for x in ["end location","dest","to","end"]):
        bp_dest_col = c
    if bp_start_col is None and any(x in cl for x in ["start time","start_time","departure","dep","time"]):
        bp_start_col = c
    if bp_activity_col is None and "activity" in cl:
        bp_activity_col = c

# Fallbacks
cols_list = list(bp.columns)
if bp_origin_col is None and len(cols_list) >= 1:
    bp_origin_col = cols_list[0]
if bp_start_col is None and len(cols_list) >= 2:
    bp_start_col = cols_list[1]
if bp_dest_col is None and len(cols_list) >= 3:
    bp_dest_col = cols_list[2]

log("BusPlan columns detected:", bp_origin_col, bp_start_col, bp_dest_col, bp_activity_col)

# Build mapping: key -> activity
bp_activity_map = {}  # map (date_time_str, origin, dest) -> activity
for i,row in bp.iterrows():
    try:
        bstart = parse_time(row.get(bp_start_col)) if bp_start_col in row.index else None
        if bstart is None:
            continue
        bstart = combine_time_with_base(bstart, BASE_DATE)
        key = (bstart.strftime("%H:%M:%S"), str(row.get(bp_origin_col)), str(row.get(bp_dest_col)))
        act = str(row.get(bp_activity_col)) if (bp_activity_col in row.index and not pd.isna(row.get(bp_activity_col))) else None
        if act is not None:
            bp_activity_map[key] = act.strip().lower()
    except Exception:
        continue

# Helper to find activity for a trip
def lookup_activity_for_trip(tr):
    k = (tr["start"].strftime("%H:%M:%S"), str(tr["origin"]), str(tr["dest"]))
    return bp_activity_map.get(k, "service")  # default to 'service' if no mapping

# ----------------------------
# Energy model helpers
# ----------------------------
battery_capacity = CONFIG["battery_capacity_kwh"] * (CONFIG["soh_percent"]/100.0)
min_allowed_soc_kwh = battery_capacity * (CONFIG["min_soc_percent_of_soh"]/100.0)

def energy_needed_for_distance_km(km, per_km=CONFIG["consumption_kwh_per_km_default"]):
    if km is None or np.isnan(km):
        return per_km * 1.0
    return per_km * km * CONFIG["consumption_variation_factor"]

def idle_energy_kwh(minutes):
    return CONFIG["idle_power_kw"] * (minutes/60.0)

def charging_energy_in_minutes(current_soc_kwh, minutes):
    if minutes <= 0:
        return 0.0, current_soc_kwh
    soc = current_soc_kwh
    energy_added = 0.0
    rem = minutes
    target_90 = battery_capacity * 0.90
    # high power to 90%
    if soc < target_90:
        need = target_90 - soc
        min_to_90 = (need / CONFIG["high_power_kw"]) * 60.0
        if rem >= min_to_90:
            energy_added += need
            soc += need
            rem -= min_to_90
        else:
            energy_added += CONFIG["high_power_kw"] * (rem/60.0)
            soc += CONFIG["high_power_kw"] * (rem/60.0)
            rem = 0
            return energy_added, min(soc, battery_capacity)
    # low power remainder
    if rem > 0:
        energy_added += CONFIG["low_power_kw"] * (rem/60.0)
        soc += CONFIG["low_power_kw"] * (rem/60.0)
    soc = min(soc, battery_capacity)
    return energy_added, soc

# ----------------------------
# Charging session tracking
# ----------------------------
charging_sessions = []  # dicts with start,end,vehicle_id

def chargers_in_interval(start, end):
    cnt = 0
    for s in charging_sessions:
        if s["start"] < end and s["end"] > start:
            cnt += 1
    return cnt

# ----------------------------
# Greedy assignment with explicit insertion of material & idle trips
# ----------------------------
vehicles = []  # each: {id, assigned_events: list of events, available_time, location, soc_kwh}
vehicle_seq = 1

# We'll build events: every event is dict with fields:
# { vehicle_id, event_type ('service','material','idle'), start, end, origin, dest, trip_row (optional), note (str) }
all_events = []

for t in trips:
    assigned = False
    for v in vehicles:
        # compute deadhead time from v.location to t.origin
        deadhead_min = estimate_deadhead_minutes(v["location"], t["origin"])
        arrive_if_leave_now = v["available_time"] + timedelta(minutes=deadhead_min)
        # can vehicle leave at v.available_time and reach before t.start?
        if arrive_if_leave_now > t["start"]:
            continue  # cannot reach in time

        # compute energy for deadhead
        dist_dh = get_distance_km(v["location"], t["origin"])
        energy_dh = energy_needed_for_distance_km(dist_dh)
        # energy for trip
        dist_trip = get_distance_km(t["origin"], t["dest"])
        energy_trip = energy_needed_for_distance_km(dist_trip)
        # idle between arrival and trip.start
        idle_minutes = (t["start"] - arrive_if_leave_now).total_seconds()/60.0
        if idle_minutes < 0:
            idle_minutes = 0.0
        energy_idle = idle_energy_kwh(idle_minutes)

        soc_after = v["soc_kwh"] - energy_dh - energy_idle - energy_trip
        if soc_after >= min_allowed_soc_kwh:
            # feasible without charging. Insert material (deadhead) event if distance > 0 or minutes>0
            # material event: leave v.available_time -> arrive_if_leave_now
            if deadhead_min > 0.1:
                ev = {
                    "vehicle_id": v["id"],
                    "event_type": "material",
                    "start": v["available_time"],
                    "end": arrive_if_leave_now,
                    "origin": v["location"],
                    "dest": t["origin"],
                    "trip_row": None,
                    "note": "deadhead"
                }
                v["assigned_events"].append(ev)
                all_events.append(ev)
            # idle event between arrival_if_leave_now and t.start (if any)
            if idle_minutes >= 0.016:  # > ~1 second
                ev = {
                    "vehicle_id": v["id"],
                    "event_type": "idle",
                    "start": arrive_if_leave_now,
                    "end": t["start"],
                    "origin": t["origin"],
                    "dest": t["origin"],
                    "trip_row": None,
                    "note": "waiting"
                }
                v["assigned_events"].append(ev)
                all_events.append(ev)
            # service event
            ev = {
                "vehicle_id": v["id"],
                "event_type": "service",
                "start": t["start"],
                "end": t["end"],
                "origin": t["origin"],
                "dest": t["dest"],
                "trip_row": t["trip_row"],
                "note": lookup_activity_for_trip(t)
            }
            v["assigned_events"].append(ev)
            all_events.append(ev)
            # update vehicle
            v["available_time"] = t["end"]
            v["location"] = t["dest"] if t["dest"] else t["origin"]
            v["soc_kwh"] = soc_after
            assigned = True
            break

        # Not enough SOC: try to charge at charging location between available_time and trip.start
        def try_charge_and_make(v):
            # deadhead to charging site
            dh_to_charge_min = estimate_deadhead_minutes(v["location"], CONFIG["charging_location"])
            arrive_charge = v["available_time"] + timedelta(minutes=dh_to_charge_min)
            # deadhead from charge to trip origin
            dh_charge_to_trip_min = estimate_deadhead_minutes(CONFIG["charging_location"], t["origin"])
            latest_charge_end = t["start"] - timedelta(minutes=dh_charge_to_trip_min)
            if latest_charge_end <= arrive_charge:
                return None
            window_min = (latest_charge_end - arrive_charge).total_seconds()/60.0
            if window_min < CONFIG["min_charging_minutes"]:
                return None

            # soc after reaching charging location
            dist1 = get_distance_km(v["location"], CONFIG["charging_location"])
            energy_d1 = energy_needed_for_distance_km(dist1)
            soc_at_charge_arrival = v["soc_kwh"] - energy_d1
            # energy needed after charging: dh to trip + trip
            dist2 = get_distance_km(CONFIG["charging_location"], t["origin"])
            energy_d2 = energy_needed_for_distance_km(dist2)
            energy_needed_post = energy_d2 + energy_needed_for_distance_km(dist_trip)
            # required soc at departure from charging location
            req_soc_at_charge_dep = min_allowed_soc_kwh + energy_needed_post
            needed_kwh = max(0.0, req_soc_at_charge_dep - soc_at_charge_arrival)
            # compute minimal minutes to charge
            if needed_kwh <= 0:
                needed_minutes = 0
            else:
                # search for minimal m in [min_charging_minutes, window_min]
                found = None
                for m in range(int(CONFIG["min_charging_minutes"]), int(math.ceil(window_min))+1):
                    added, new_soc = charging_energy_in_minutes(soc_at_charge_arrival, m)
                    if added + 1e-6 >= needed_kwh:
                        found = m
                        break
                if found is None:
                    return None
                needed_minutes = found

            # check charger capacity in [arrive_charge, arrive_charge+needed_minutes]
            charge_start = arrive_charge
            charge_end = arrive_charge + timedelta(minutes=needed_minutes)
            if chargers_in_interval(charge_start, charge_end) >= CONFIG["charging_station_capacity"]:
                return None

            # compute SOC after charging and after travel to trip origin and trip itself
            added, soc_after_charge = charging_energy_in_minutes(soc_at_charge_arrival, needed_minutes)
            soc_at_trip_start = soc_after_charge - energy_d2
            soc_after_trip_local = soc_at_trip_start - energy_needed_for_distance_km(dist_trip)
            if soc_after_trip_local < min_allowed_soc_kwh - 1e-6:
                return None

            # Build planned events: material (to charge), idle (charging), material (charge->trip origin), service
            events = []
            if dh_to_charge_min > 0.1:
                events.append({
                    "vehicle_id": v["id"],
                    "event_type": "material",
                    "start": v["available_time"],
                    "end": arrive_charge,
                    "origin": v["location"],
                    "dest": CONFIG["charging_location"],
                    "trip_row": None,
                    "note": "deadhead to charge"
                })
            # charging idle event (mark as idle but note charging)
            events.append({
                "vehicle_id": v["id"],
                "event_type": "idle",
                "start": charge_start,
                "end": charge_end,
                "origin": CONFIG["charging_location"],
                "dest": CONFIG["charging_location"],
                "trip_row": None,
                "note": "charging"
            })
            # material from charge to trip origin - arrival should be <= trip.start
            depart_charge = charge_end
            arrive_trip_origin = depart_charge + timedelta(minutes=dh_charge_to_trip_min)
            if dh_charge_to_trip_min > 0.1:
                events.append({
                    "vehicle_id": v["id"],
                    "event_type": "material",
                    "start": depart_charge,
                    "end": arrive_trip_origin,
                    "origin": CONFIG["charging_location"],
                    "dest": t["origin"],
                    "trip_row": None,
                    "note": "deadhead from charge"
                })
            # small idle if early
            idle_minutes_local = (t["start"] - arrive_trip_origin).total_seconds()/60.0
            if idle_minutes_local > 0.016:
                events.append({
                    "vehicle_id": v["id"],
                    "event_type": "idle",
                    "start": arrive_trip_origin,
                    "end": t["start"],
                    "origin": t["origin"],
                    "dest": t["origin"],
                    "trip_row": None,
                    "note": "waiting after charge"
                })
            # service
            events.append({
                "vehicle_id": v["id"],
                "event_type": "service",
                "start": t["start"],
                "end": t["end"],
                "origin": t["origin"],
                "dest": t["dest"],
                "trip_row": t["trip_row"],
                "note": lookup_activity_for_trip(t)
            })
            # return plan with soc_after_trip_local and session info
            plan = {
                "events": events,
                "charge_session": {"vehicle_id": v["id"], "start": charge_start, "end": charge_end},
                "soc_after_trip": soc_after_trip_local,
                "new_available_time": t["end"],
                "new_location": t["dest"] if t["dest"] else t["origin"]
            }
            return plan

        plan = try_charge_and_make(v)
        if plan is not None:
            # commit events and charging session
            for ev in plan["events"]:
                v["assigned_events"].append(ev)
                all_events.append(ev)
            charging_sessions.append(plan["charge_session"])
            v["available_time"] = plan["new_available_time"]
            v["location"] = plan["new_location"]
            v["soc_kwh"] = plan["soc_after_trip"]
            assigned = True
            break

    if not assigned:
        # try to create new vehicle starting at charging_location (assume available earlier)
        init_soc = battery_capacity * (CONFIG["initial_soc_percent"]/100.0)
        dist_from_depot = get_distance_km(CONFIG["charging_location"], t["origin"])
        energy_to_origin = energy_needed_for_distance_km(dist_from_depot)
        dist_trip = get_distance_km(t["origin"], t["dest"])
        energy_trip = energy_needed_for_distance_km(dist_trip)
        soc_after = init_soc - energy_to_origin - energy_trip
        if soc_after >= min_allowed_soc_kwh:
            # new vehicle travels from depot -> trip origin (material), maybe small idle, then service
            v = {
                "id": vehicle_seq,
                "assigned_events": [],
                "available_time": t["end"],
                "location": t["dest"] if t["dest"] else t["origin"],
                "soc_kwh": soc_after
            }
            # material from depot to origin
            dh_minutes = estimate_deadhead_minutes(CONFIG["charging_location"], t["origin"])
            depart_depot = datetime.combine(BASE_DATE.date(), datetime.min.time())  # effectively early
            arrive_origin = depart_depot + timedelta(minutes=dh_minutes)
            # if arrive_origin > t.start, adapt depart_depot earlier; keeping simple: assume possible
            ev_mat = {
                "vehicle_id": v["id"],
                "event_type": "material",
                "start": depart_depot,
                "end": arrive_origin,
                "origin": CONFIG["charging_location"],
                "dest": t["origin"],
                "trip_row": None,
                "note": "initial deadhead from depot"
            }
            v["assigned_events"].append(ev_mat); all_events.append(ev_mat)
            if arrive_origin < t["start"]:
                ev_idle = {
                    "vehicle_id": v["id"],
                    "event_type": "idle",
                    "start": arrive_origin,
                    "end": t["start"],
                    "origin": t["origin"],
                    "dest": t["origin"],
                    "trip_row": None,
                    "note": "waiting before first service"
                }
                v["assigned_events"].append(ev_idle); all_events.append(ev_idle)
            ev_service = {
                "vehicle_id": v["id"],
                "event_type": "service",
                "start": t["start"],
                "end": t["end"],
                "origin": t["origin"],
                "dest": t["dest"],
                "trip_row": t["trip_row"],
                "note": lookup_activity_for_trip(t)
            }
            v["assigned_events"].append(ev_service); all_events.append(ev_service)
            vehicles.append(v)
            vehicle_seq += 1
            assigned = True
        else:
            # attempt initial charge at depot before starting
            required_after = min_allowed_soc_kwh + energy_to_origin + energy_trip
            curr = init_soc
            need_kwh = max(0.0, required_after - curr)
            # find minutes needed
            found = None
            for m in range(CONFIG["min_charging_minutes"], 12*60):
                added, new_soc = charging_energy_in_minutes(curr, m)
                if added >= need_kwh - 1e-6:
                    found = m
                    break
            if found is None:
                log("Cannot schedule new vehicle for trip at", t["start"], "due to energy limits.")
                continue
            # schedule charging session (we won't block capacity here for initial vehicles)
            charge_start = datetime.combine(BASE_DATE.date(), datetime.min.time())  # early
            charge_end = charge_start + timedelta(minutes=found)
            charging_sessions.append({"vehicle_id": vehicle_seq, "start": charge_start, "end": charge_end})
            added, soc_after_charge = charging_energy_in_minutes(curr, found)
            # then deadhead to origin
            dh_minutes = estimate_deadhead_minutes(CONFIG["charging_location"], t["origin"])
            depart = charge_end
            arrive_origin = depart + timedelta(minutes=dh_minutes)
            # create vehicle and events
            v = {
                "id": vehicle_seq,
                "assigned_events": [],
                "available_time": t["end"],
                "location": t["dest"] if t["dest"] else t["origin"],
                "soc_kwh": soc_after_charge - energy_needed_for_distance_km(get_distance_km(CONFIG["charging_location"], t["origin"])) - energy_needed_for_distance_km(get_distance_km(t["origin"], t["dest"]))
            }
            ev_charge = {"vehicle_id": v["id"], "event_type": "idle", "start": charge_start, "end": charge_end, "origin": CONFIG["charging_location"], "dest": CONFIG["charging_location"], "trip_row": None, "note": "initial charging"}
            v["assigned_events"].append(ev_charge); all_events.append(ev_charge)
            ev_mat = {"vehicle_id": v["id"], "event_type": "material", "start": charge_end, "end": arrive_origin, "origin": CONFIG["charging_location"], "dest": t["origin"], "trip_row": None, "note":"initial deadhead"}
            v["assigned_events"].append(ev_mat); all_events.append(ev_mat)
            if arrive_origin < t["start"]:
                ev_idle = {"vehicle_id": v["id"], "event_type":"idle", "start": arrive_origin, "end": t["start"], "origin": t["origin"], "dest": t["origin"], "trip_row": None, "note":"waiting"}
                v["assigned_events"].append(ev_idle); all_events.append(ev_idle)
            ev_service = {"vehicle_id": v["id"], "event_type":"service","start": t["start"], "end": t["end"], "origin": t["origin"], "dest": t["dest"], "trip_row": t["trip_row"], "note": lookup_activity_for_trip(t)}
            v["assigned_events"].append(ev_service); all_events.append(ev_service)
            vehicles.append(v)
            vehicle_seq += 1
            assigned = True

# ----------------------------
# Aggregate and save outputs
# ----------------------------
log("Vehicles used:", len(vehicles))
# Build AssignedTrips (service only) and AllTrips (service/material/idle)
assigned_rows = []
for ev in all_events:
    if ev["event_type"] == "service":
        assigned_rows.append({
            "vehicle_id": ev["vehicle_id"],
            "start": ev["start"],
            "end": ev["end"],
            "origin": ev["origin"],
            "dest": ev["dest"],
            "trip_row": ev["trip_row"],
            "activity": ev["note"]
        })

assigned_df = pd.DataFrame(assigned_rows)
# sequence per vehicle for AssignedTrips
if not assigned_df.empty:
    assigned_df = assigned_df.sort_values(["vehicle_id","start"]).reset_index(drop=True)
    assigned_df["sequence"] = assigned_df.groupby("vehicle_id").cumcount()+1

all_rows = []
for ev in all_events:
    all_rows.append({
        "vehicle_id": ev["vehicle_id"],
        "event_type": ev["event_type"],
        "start": ev["start"],
        "end": ev["end"],
        "origin": ev["origin"],
        "dest": ev["dest"],
        "trip_row": ev["trip_row"],
        "note": ev["note"]
    })
all_df = pd.DataFrame(all_rows)
if not all_df.empty:
    all_df = all_df.sort_values(["vehicle_id","start"]).reset_index(drop=True)
    all_df["sequence"] = all_df.groupby("vehicle_id").cumcount()+1

veh_summary = []
for v in vehicles:
    veh_summary.append({
        "vehicle_id": v["id"],
        "num_events": len(v["assigned_events"]),
        "final_location": v["location"],
        "available_time": v["available_time"],
        "final_soc_kwh": round(v.get("soc_kwh", 0.0),3),
        "final_soc_pct": round((v.get("soc_kwh",0.0)/battery_capacity)*100.0,2) if battery_capacity>0 else None
    })
veh_summary_df = pd.DataFrame(veh_summary)

out_xlsx = os.path.join(OUT_DIR, "improved_bus_schedule_with_activities.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    assigned_df.to_excel(w, sheet_name="AssignedTrips_ServiceOnly", index=False)
    all_df.to_excel(w, sheet_name="AllTrips_ServiceMaterialIdle", index=False)
    veh_summary_df.to_excel(w, sheet_name="VehicleSummary", index=False)
    try:
        timetable_df.to_excel(w, sheet_name="OriginalTimetable", index=False)
    except:
        pass
    try:
        dist_numeric.to_excel(w, sheet_name="DistanceMatrix", index=True)
    except:
        pass
    try:
        busplan_df.head(200).to_excel(w, sheet_name="BusPlanningSample", index=False)
    except:
        pass

log("Saved Excel to:", out_xlsx)

# ----------------------------
# Gantt chart (interactive) - color by event_type
# ----------------------------
if all_df.empty:
    log("No events to plot.")
else:
    gantt_df = all_df.copy()
    gantt_df["start_dt"] = pd.to_datetime(gantt_df["start"])
    gantt_df["end_dt"] = pd.to_datetime(gantt_df["end"])
    # label for hover
    gantt_df["label"] = gantt_df.apply(lambda r: f"Veh {r.vehicle_id} | {r.event_type} | {r.origin}->{r.dest} | {r.start_dt.time()} - {r.end_dt.time()} | {r.note}", axis=1)
    # map colors: service/material/idle
    color_map = {"service":"Service", "material":"Material (deadhead)", "idle":"Idle/Charging"}
    gantt_df["activity_label"] = gantt_df["event_type"].map(color_map).fillna(gantt_df["event_type"])
    fig = px.timeline(gantt_df, x_start="start_dt", x_end="end_dt", y="vehicle_id", color="activity_label",
                      hover_data=["label","sequence"], title="Bus Schedule: service/material/idle")
    fig.update_yaxes(title="Vehicle ID", autorange="reversed")
    fig.update_layout(height=700)
    out_html = os.path.join(OUT_DIR, "gantt_with_activities.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    log("Saved Gantt (HTML) to:", out_html)

log("Done. Outputs in folder:", OUT_DIR)
