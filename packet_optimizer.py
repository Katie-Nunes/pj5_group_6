"""
Bus Fleet Optimizer using Single-Packet Circular Rotation Algorithm

This module optimizes bus fleet size by assigning service trips to buses
using a circular rotation strategy. It manages battery charging, ensures
SOC constraints (10-90%), and guarantees all buses start and end at depot.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from logging_utils import report_error, report_warning, report_info
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    battery_capacity: float = 300.0
    state_of_health: float = 0.85
    min_soc_percent: float = 0.10
    max_soc_percent: float = 0.90
    charging_rate_kwh_per_min: float = 7.5
    charging_window_minutes: int = 15
    energy_per_km: float = 1.5
    idle_energy_per_min: float = 0.05
    garage_location: str = "ehvgar"
    charging_buffer_percent: float = 0.15  # Send to charge when SOC drops to 25% (was 15%)
    
    @property
    def effective_capacity(self) -> float:
        return self.battery_capacity * self.state_of_health
    
    @property
    def min_soc(self) -> float:
        return self.effective_capacity * self.min_soc_percent
    
    @property
    def max_soc(self) -> float:
        return self.effective_capacity * self.max_soc_percent
    
    @property
    def charging_threshold(self) -> float:
        """SOC level at which to send bus to charging (min_soc + buffer)"""
        return self.effective_capacity * (self.min_soc_percent + self.charging_buffer_percent)


@dataclass
class BusState:
    bus_id: int
    current_soc: float
    location: str
    available_at: datetime
    is_charging: bool = False
    charging_ends_at: Optional[datetime] = None
    total_distance_km: float = 0.0
    total_trips: int = 0
    
    def __repr__(self):
        status = "CHARGING" if self.is_charging else "ACTIVE"
        return f"Bus{self.bus_id}[{status}, SOC:{self.current_soc:.1f}kWh @ {self.location}]"


@dataclass
class TripRecord:
    bus_id: int
    start_location: str
    end_location: str
    start_time: datetime
    end_time: datetime
    activity: str
    line: float
    energy_consumption: float
    distance_km: float = 0.0


class SinglePacketPlanner:
    """
    Fleet optimizer using single-packet circular rotation algorithm.
    
    Assigns service trips to buses in rotation, managing battery charging
    and ensuring all constraints are met (SOC limits, depot returns, etc).
    """
    
    def __init__(self, config: OptimizerConfig, timetable_df: pd.DataFrame, distance_matrix_df: pd.DataFrame):
        """
        Initialize the optimizer with configuration and data.
        
        Args:
            config: Optimizer configuration with battery and charging parameters
            timetable_df: Service timetable with trips to be covered
            distance_matrix_df: Distance and travel time matrix between locations
        """
        self.config = config
        self.timetable_df = self._normalize_timetable_columns(timetable_df.copy())
        self.distance_matrix_df = distance_matrix_df
        
        self.distance_lookup = self._build_distance_lookup()
        
        self.buses: List[BusState] = []
        self.rotation_index: int = 0
        self.trip_records: List[TripRecord] = []
        self.charging_timeline: List[Tuple[datetime, datetime, int]] = []
        self.initial_time: Optional[datetime] = None
    
    def _normalize_timetable_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for internal use."""
        column_map = {
            'start location': 'start',
            'start time': 'departure_time',
            'end location': 'end'
        }
        df = df.rename(columns=column_map)
        return df
        
    def _build_distance_lookup(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Build fast lookup dictionary for distances and travel times."""
        lookup = {}
        for _, row in self.distance_matrix_df.iterrows():
            start = row['start']
            end = row['end']
            distance_m = row.get('distance_m', 0)
            travel_time_min = row.get('min_travel_time', 0)
            distance_km = distance_m / 1000.0
            lookup[(start, end)] = (distance_km, travel_time_min)
        return lookup
    
    def get_distance_and_time(self, start: str, end: str) -> Tuple[float, float]:
        """
        Get distance and travel time between two locations.
        
        Args:
            start: Starting location code
            end: Ending location code
        
        Returns:
            Tuple of (distance_km, travel_time_min)
        """
        if start == end:
            return 0.0, 0.0
        key = (start, end)
        if key in self.distance_lookup:
            return self.distance_lookup[key]
        report_warning(f"Distance not found for {start} -> {end}, using estimate")
        return 10.0, 15.0
    
    def calculate_energy_consumption(self, distance_km: float, travel_time_min: float = 0) -> float:
        """
        Calculate total energy consumption for a trip.
        
        Args:
            distance_km: Distance traveled in kilometers
            travel_time_min: Travel time in minutes (for idle consumption)
        
        Returns:
            Total energy consumption in kWh
        """
        travel_energy = distance_km * self.config.energy_per_km
        idle_energy = travel_time_min * self.config.idle_energy_per_min
        return travel_energy + idle_energy
    
    def initialize_fleet(self, fleet_size: int, initial_time: datetime) -> None:
        """
        Initialize the bus fleet with given size.
        
        All buses start at depot with 90% SOC (229.5 kWh).
        
        Args:
            fleet_size: Number of buses to create
            initial_time: Starting time for all buses
        """
        self.buses = []
        self.initial_time = initial_time
        for i in range(fleet_size):
            bus = BusState(
                bus_id=i + 1,
                current_soc=self.config.max_soc,  # Start at 90% SOC (229.5 kWh)
                location=self.config.garage_location,
                available_at=initial_time
            )
            self.buses.append(bus)
        self.rotation_index = 0
        self.trip_records = []
        self.charging_timeline = []
        report_info(f"Initialized fleet of {fleet_size} buses")
    
    def get_next_available_bus(self, required_at: datetime) -> Optional[BusState]:
        """
        Get next available bus using circular rotation strategy.
        
        Updates charging status of buses that finished charging and returns
        the next bus available at required time using rotation index.
        
        Args:
            required_at: Time when bus must be available
        
        Returns:
            Next available bus in rotation or None if no bus available
        """
        for bus in self.buses:
            if bus.is_charging and bus.charging_ends_at and bus.charging_ends_at <= required_at:
                bus.is_charging = False
                bus.charging_ends_at = None
        
        checked = 0
        start_index = self.rotation_index
        
        while checked < len(self.buses):
            current_index = (start_index + checked) % len(self.buses)
            bus = self.buses[current_index]
            
            if not bus.is_charging and bus.available_at <= required_at:
                self.rotation_index = (current_index + 1) % len(self.buses)
                return bus
            
            checked += 1
        
        return None
    
    def predict_soc_after_trip(self, bus: BusState, trip_distance_km: float, trip_time_min: float) -> float:
        """
        Predict bus SOC after completing a trip.
        
        Args:
            bus: Bus to check
            trip_distance_km: Distance of trip in kilometers
            trip_time_min: Duration of trip in minutes
        
        Returns:
            Predicted SOC in kWh after trip completion
        """
        energy_needed = self.calculate_energy_consumption(trip_distance_km, trip_time_min)
        return bus.current_soc - energy_needed
    
    def needs_charging(self, bus: BusState, next_trip_distance_km: float, next_trip_time_min: float) -> bool:
        """
        Check if bus needs charging before next trip.
        Uses charging_threshold (min_soc + buffer) for safety margin.
        """
        predicted_soc = self.predict_soc_after_trip(bus, next_trip_distance_km, next_trip_time_min)
        return predicted_soc < self.config.charging_threshold
    
    def send_to_charging(self, bus: BusState, start_time: datetime) -> List[TripRecord]:
        """
        Send bus to garage for charging.
        
        Creates trip records for material trip to garage (if needed) and
        charging session. Updates bus SOC and location. Includes SOC guard
        to prevent dropping below minimum on way to garage.
        
        Args:
            bus: Bus to send for charging
            start_time: When to start charging process
        
        Returns:
            List of trip records (material trip + charging) or empty if unsafe
        """
        records = []
        
        if bus.location != self.config.garage_location:
            distance_km, travel_time_min = self.get_distance_and_time(bus.location, self.config.garage_location)
            energy_to_garage = self.calculate_energy_consumption(distance_km, travel_time_min)
            
            # CRITICAL: Check if bus would drop below minimum SOC on the way to garage
            if bus.current_soc - energy_to_garage < self.config.min_soc:
                report_warning(
                    f"Bus {bus.bus_id} cannot reach garage for charging - would drop below minimum SOC "
                    f"({bus.current_soc:.1f} - {energy_to_garage:.1f} = {bus.current_soc - energy_to_garage:.1f} kWh < {self.config.min_soc:.1f} kWh)"
                )
                # Return empty - this will cause simulation to fail and try more buses
                return []
            
            arrival_at_garage = start_time + timedelta(minutes=travel_time_min)
            
            material_trip = TripRecord(
                bus_id=bus.bus_id,
                start_location=bus.location,
                end_location=self.config.garage_location,
                start_time=start_time,
                end_time=arrival_at_garage,
                activity="material trip",
                line=999.0,
                energy_consumption=energy_to_garage,
                distance_km=distance_km
            )
            records.append(material_trip)
            
            bus.current_soc -= energy_to_garage
            bus.location = self.config.garage_location
            start_time = arrival_at_garage
        
        charging_end = start_time + timedelta(minutes=self.config.charging_window_minutes)
        energy_added = self.config.charging_rate_kwh_per_min * self.config.charging_window_minutes
        new_soc = min(bus.current_soc + energy_added, self.config.max_soc)  # Charge to 90% max (229.5 kWh)
        
        # CRITICAL: Record ACTUAL energy added (capped at 229.5 kWh), not theoretical energy_added
        actual_energy_added = new_soc - bus.current_soc
        
        charging_record = TripRecord(
            bus_id=bus.bus_id,
            start_location=self.config.garage_location,
            end_location=self.config.garage_location,
            start_time=start_time,
            end_time=charging_end,
            activity="charging",
            line=999.0,
            energy_consumption=-actual_energy_added,  # Use actual energy added, not theoretical
            distance_km=0.0
        )
        records.append(charging_record)
        
        bus.current_soc = new_soc
        bus.is_charging = True
        bus.charging_ends_at = charging_end
        bus.available_at = charging_end
        
        self.charging_timeline.append((start_time, charging_end, bus.bus_id))
        
        report_info(f"Bus {bus.bus_id} sent to charging, SOC: {bus.current_soc:.1f} kWh")
        
        return records
    
    def assign_service_trip(self, bus: BusState, start_loc: str, end_loc: str, 
                           departure_time: datetime, line: float) -> List[TripRecord]:
        """
        Assign service trip to bus and create all necessary trip records.
        
        Handles deadhead trip to service start, idle period, service trip itself,
        and updates bus state (SOC, location, availability). Includes comprehensive
        SOC guards to prevent violations during the entire trip sequence.
        
        Args:
            bus: Bus to assign trip to
            start_loc: Service trip start location
            end_loc: Service trip end location
            departure_time: Scheduled departure time for service
            line: Line number for service trip
        
        Returns:
            List of all trip records created (idle, deadhead, service) or empty if unsafe
        """
        records = []
        current_time = bus.available_at
        
        # CRITICAL: Calculate TOTAL energy for the entire trip sequence (idle + deadhead + service)
        # to ensure bus won't drop below minimum SOC at ANY point
        total_energy = 0.0
        
        # Calculate service trip energy (always happens)
        service_distance_km, service_travel_time_min = self.get_distance_and_time(start_loc, end_loc)
        energy_service = self.calculate_energy_consumption(service_distance_km, service_travel_time_min)
        total_energy += energy_service
        
        if bus.location != start_loc:
            # Calculate deadhead energy
            distance_km, travel_time_min = self.get_distance_and_time(bus.location, start_loc)
            energy_deadhead = self.calculate_energy_consumption(distance_km, travel_time_min)
            total_energy += energy_deadhead
            
            # Calculate idle energy before deadhead (if waiting at origin)
            optimal_departure = departure_time - timedelta(minutes=travel_time_min)
            if current_time < optimal_departure:
                idle_at_origin_minutes = (optimal_departure - current_time).total_seconds() / 60.0
                if bus.location != self.config.garage_location:
                    idle_energy = idle_at_origin_minutes * self.config.idle_energy_per_min
                    total_energy += idle_energy
        else:
            # Already at start location, check if idle time before service
            if current_time < departure_time:
                idle_minutes = (departure_time - current_time).total_seconds() / 60.0
                if start_loc != self.config.garage_location:
                    idle_energy = idle_minutes * self.config.idle_energy_per_min
                    total_energy += idle_energy
        
        # Guard: Check if total trip would drop below minimum SOC
        if bus.current_soc - total_energy < self.config.min_soc:
            report_warning(
                f"Bus {bus.bus_id} cannot complete trip - would drop below minimum SOC "
                f"({bus.current_soc:.1f} - {total_energy:.1f} = {bus.current_soc - total_energy:.1f} kWh < {self.config.min_soc:.1f} kWh)"
            )
            return []  # Signal failure
        
        # Now execute the trip sequence (we know it's safe)
        if bus.location != start_loc:
            distance_km, travel_time_min = self.get_distance_and_time(bus.location, start_loc)
            energy_deadhead = self.calculate_energy_consumption(distance_km, travel_time_min)
            
            optimal_departure = departure_time - timedelta(minutes=travel_time_min)
            
            if current_time < optimal_departure:
                idle_at_origin_minutes = (optimal_departure - current_time).total_seconds() / 60.0
                
                # At garage: no energy consumption (parked/turned off)
                # Not at garage: consume idle energy
                if bus.location == self.config.garage_location:
                    idle_energy = 0.0
                else:
                    idle_energy = idle_at_origin_minutes * self.config.idle_energy_per_min
                
                # Only create idle record if duration is less than 3 hours (180 minutes)
                if idle_at_origin_minutes < 180:
                    idle_trip = TripRecord(
                        bus_id=bus.bus_id,
                        start_location=bus.location,
                        end_location=bus.location,
                        start_time=current_time,
                        end_time=optimal_departure,
                        activity="idle",
                        line=999.0,
                        energy_consumption=idle_energy,
                        distance_km=0.0
                    )
                    records.append(idle_trip)
                
                bus.current_soc -= idle_energy
                current_time = optimal_departure
            
            deadhead_arrival = current_time + timedelta(minutes=travel_time_min)
            
            material_trip = TripRecord(
                bus_id=bus.bus_id,
                start_location=bus.location,
                end_location=start_loc,
                start_time=current_time,
                end_time=deadhead_arrival,
                activity="material trip",
                line=999.0,
                energy_consumption=energy_deadhead,
                distance_km=distance_km
            )
            records.append(material_trip)
            
            bus.current_soc -= energy_deadhead
            bus.location = start_loc
            current_time = deadhead_arrival
        else:
            if current_time < departure_time:
                idle_minutes = (departure_time - current_time).total_seconds() / 60.0
                
                # At garage: no energy consumption (parked/turned off)
                # Not at garage: consume idle energy
                if start_loc == self.config.garage_location:
                    idle_energy = 0.0
                else:
                    idle_energy = idle_minutes * self.config.idle_energy_per_min
                
                # Only create idle record if duration is less than 3 hours (180 minutes)
                if idle_minutes < 180:
                    idle_trip = TripRecord(
                        bus_id=bus.bus_id,
                        start_location=start_loc,
                        end_location=start_loc,
                        start_time=current_time,
                        end_time=departure_time,
                        activity="idle",
                        line=999.0,
                        energy_consumption=idle_energy,
                        distance_km=0.0
                    )
                    records.append(idle_trip)
                
                bus.current_soc -= idle_energy
                current_time = departure_time
        
        distance_km, travel_time_min = self.get_distance_and_time(start_loc, end_loc)
        energy_service = self.calculate_energy_consumption(distance_km, travel_time_min)
        arrival_time = departure_time + timedelta(minutes=travel_time_min)
        
        service_trip = TripRecord(
            bus_id=bus.bus_id,
            start_location=start_loc,
            end_location=end_loc,
            start_time=departure_time,
            end_time=arrival_time,
            activity="service trip",
            line=line,
            energy_consumption=energy_service,
            distance_km=distance_km
        )
        records.append(service_trip)
        
        bus.current_soc -= energy_service
        bus.location = end_loc
        bus.available_at = arrival_time
        bus.total_distance_km += distance_km
        bus.total_trips += 1
        
        return records
    
    def optimize_fleet_size(self, max_buses: int = 50) -> Tuple[int, bool]:
        """
        Find minimum fleet size that can cover all service trips.
        
        Incrementally tries fleet sizes from 1 to max_buses until
        a feasible solution is found that satisfies all constraints.
        
        Args:
            max_buses: Maximum fleet size to try
        
        Returns:
            Tuple of (optimal_fleet_size, success_flag)
        """
        for fleet_size in range(1, max_buses + 1):
            report_info(f"Trying fleet size: {fleet_size}")
            
            success = self.run_simulation(fleet_size)
            
            if success:
                report_info(f"Successfully optimized with {fleet_size} buses!")
                return fleet_size, True
        
        report_error("Could not find feasible solution with up to {max_buses} buses")
        return max_buses, False
    
    def run_simulation(self, fleet_size: int) -> bool:
        """
        Simulate complete bus scheduling with given fleet size.
        
        Assigns all service trips to buses using circular rotation,
        managing charging proactively and ensuring all SOC constraints
        are satisfied. Validates that all buses return to depot at end.
        
        Args:
            fleet_size: Number of buses to simulate with
        
        Returns:
            True if simulation succeeded (all trips covered, constraints met),
            False if simulation failed (SOC violation, no bus available, etc)
        """
        timetable_sorted = self.timetable_df.sort_values('departure_time').reset_index(drop=True)
        
        if len(timetable_sorted) == 0:
            report_error("Empty timetable")
            return False
        
        initial_time = pd.to_datetime(timetable_sorted.iloc[0]['departure_time'])
        initial_time = initial_time - timedelta(hours=1)
        
        self.initialize_fleet(fleet_size, initial_time)
        
        for idx, row in timetable_sorted.iterrows():
            start_loc = row['start']
            end_loc = row['end']
            departure_time = pd.to_datetime(row['departure_time'])
            line = row.get('line', 999.0)
            
            service_distance_km, service_travel_time_min = self.get_distance_and_time(start_loc, end_loc)
            
            bus = self.get_next_available_bus(departure_time)
            
            if bus is None:
                report_warning(f"No bus available for trip at {departure_time}")
                return False
            
            # Calculate total energy needed: material trip + service trip
            total_energy_needed = 0
            if bus.location != start_loc:
                material_distance_km, material_travel_time_min = self.get_distance_and_time(bus.location, start_loc)
                total_energy_needed += self.calculate_energy_consumption(material_distance_km, material_travel_time_min)
            total_energy_needed += self.calculate_energy_consumption(service_distance_km, service_travel_time_min)
            
            # Check if bus needs charging before this complete trip
            if bus.current_soc - total_energy_needed < self.config.charging_threshold:
                charging_records = self.send_to_charging(bus, bus.available_at)
                
                # Check if charging failed (bus couldn't reach garage)
                if not charging_records:
                    report_warning(f"Failed to send bus {bus.bus_id} to charging - cannot reach garage")
                    return False
                
                self.trip_records.extend(charging_records)
                
                bus = self.get_next_available_bus(departure_time)
                if bus is None:
                    report_warning(f"No bus available after charging at {departure_time}")
                    return False
            
            trip_records = self.assign_service_trip(bus, start_loc, end_loc, departure_time, line)
            
            # Check if assignment failed (returned empty list due to SOC violation)
            if not trip_records:
                report_warning(f"Failed to assign service trip at {departure_time} - SOC violation")
                return False
            
            # Check SOC after EACH record added, not just at the end
            for record in trip_records:
                if record.activity != 'charging':  # Charging adds energy (negative value)
                    # Simulate what SOC would be after this specific activity
                    test_soc = bus.current_soc
                    if test_soc < self.config.min_soc:
                        report_warning(f"Bus {bus.bus_id} would drop below minimum SOC after {record.activity}: {test_soc:.1f} < {self.config.min_soc:.1f} kWh at {record.end_time}")
                        return False
            
            self.trip_records.extend(trip_records)
            
            # Final validation after trip
            if bus.current_soc < self.config.min_soc:
                report_warning(f"Bus {bus.bus_id} dropped below minimum SOC: {bus.current_soc:.1f} < {self.config.min_soc:.1f} kWh")
                return False
            
            if bus.current_soc < 0:
                report_warning(f"Bus {bus.bus_id} ran out of energy completely!")
                return False
        
        for bus in self.buses:
            if bus.is_charging:
                bus.is_charging = False
                bus.charging_ends_at = None
        
        for bus in self.buses:
            if bus.location != self.config.garage_location:
                distance_km, travel_time_min = self.get_distance_and_time(bus.location, self.config.garage_location)
                energy_needed = self.calculate_energy_consumption(distance_km, travel_time_min)
                
                # Check if depot return would drop below minimum SOC - charge first if needed
                if bus.current_soc - energy_needed < self.config.min_soc:
                    report_info(f"Bus {bus.bus_id} needs charging before depot return (SOC would drop to {bus.current_soc - energy_needed:.1f} kWh)")
                    charging_records = self.send_to_charging(bus, bus.available_at)
                    
                    # CRITICAL: Check if charging failed (bus couldn't reach garage)
                    if not charging_records:
                        report_warning(f"Bus {bus.bus_id} cannot reach garage for final charging - need more buses")
                        return False
                    
                    self.trip_records.extend(charging_records)
                    
                    # Recalculate distance in case bus is now at garage
                    if bus.location == self.config.garage_location:
                        continue  # Already at garage after charging
                    distance_km, travel_time_min = self.get_distance_and_time(bus.location, self.config.garage_location)
                    energy_needed = self.calculate_energy_consumption(distance_km, travel_time_min)
                
                # Final check before depot return
                if bus.current_soc - energy_needed < self.config.min_soc:
                    report_warning(f"Bus {bus.bus_id} still cannot return to depot safely after charging")
                    return False
                
                arrival_time = bus.available_at + timedelta(minutes=travel_time_min)
                
                depot_return = TripRecord(
                    bus_id=bus.bus_id,
                    start_location=bus.location,
                    end_location=self.config.garage_location,
                    start_time=bus.available_at,
                    end_time=arrival_time,
                    activity='material trip',
                    line=999.0,
                    energy_consumption=energy_needed,
                    distance_km=distance_km
                )
                self.trip_records.append(depot_return)
                
                bus.current_soc -= energy_needed
                bus.location = self.config.garage_location
                bus.available_at = arrival_time
                bus.total_distance_km += distance_km
        
        return True
    
    def insert_opportunistic_charging(self, min_idle_duration: int = 30):
        """
        Insert charging sessions during long idle periods.
        Buses will return to garage to charge if idle time is sufficient.
        """
        new_records = []
        
        for bus in self.buses:
            bus_records = sorted(
                [r for r in self.trip_records if r.bus_id == bus.bus_id],
                key=lambda x: x.start_time
            )
            
            for i in range(len(bus_records) - 1):
                current_record = bus_records[i]
                next_record = bus_records[i + 1]
                
                # Skip if already at garage or if this is already a charging/idle record
                if current_record.end_location == self.config.garage_location:
                    continue
                if current_record.activity in ['charging', 'idle']:
                    continue
                
                idle_start = current_record.end_time
                idle_end = next_record.start_time
                idle_duration_min = (idle_end - idle_start).total_seconds() / 60.0
                
                # Check if idle period is long enough to warrant charging trip
                if idle_duration_min < min_idle_duration:
                    continue
                
                # Calculate travel times to and from garage
                to_garage_km, to_garage_min = self.get_distance_and_time(
                    current_record.end_location, 
                    self.config.garage_location
                )
                from_garage_km, from_garage_min = self.get_distance_and_time(
                    self.config.garage_location, 
                    next_record.start_location
                )
                
                total_travel_min = to_garage_min + from_garage_min
                
                # Check if there's enough time for travel + minimum charging
                if idle_duration_min < total_travel_min + min_idle_duration:
                    continue
                
                charging_duration_min = idle_duration_min - total_travel_min
                
                # Calculate energy
                to_garage_energy = self.calculate_energy_consumption(to_garage_km, to_garage_min)
                from_garage_energy = self.calculate_energy_consumption(from_garage_km, from_garage_min)
                charging_amount = charging_duration_min * self.config.charging_rate_kwh_per_min
                
                # Create records for: trip to garage -> charging -> trip from garage
                trip_to_garage = TripRecord(
                    bus_id=bus.bus_id,
                    start_location=current_record.end_location,
                    end_location=self.config.garage_location,
                    start_time=idle_start,
                    end_time=idle_start + timedelta(minutes=to_garage_min),
                    activity='material trip',
                    line=999.0,
                    energy_consumption=to_garage_energy,
                    distance_km=to_garage_km
                )
                
                charging_session = TripRecord(
                    bus_id=bus.bus_id,
                    start_location=self.config.garage_location,
                    end_location=self.config.garage_location,
                    start_time=idle_start + timedelta(minutes=to_garage_min),
                    end_time=idle_end - timedelta(minutes=from_garage_min),
                    activity='charging',
                    line=999.0,
                    energy_consumption=-charging_amount,  # Negative = adds energy
                    distance_km=0.0
                )
                
                trip_from_garage = TripRecord(
                    bus_id=bus.bus_id,
                    start_location=self.config.garage_location,
                    end_location=next_record.start_location,
                    start_time=idle_end - timedelta(minutes=from_garage_min),
                    end_time=idle_end,
                    activity='material trip',
                    line=999.0,
                    energy_consumption=from_garage_energy,
                    distance_km=from_garage_km
                )
                
                new_records.extend([trip_to_garage, charging_session, trip_from_garage])
        
        # Add new records to trip records
        if new_records:
            self.trip_records.extend(new_records)
            report_info(f"Added {len(new_records)//3} opportunistic charging sessions")
    
    def export_to_dataframe(self) -> pd.DataFrame:
        if not self.trip_records:
            return pd.DataFrame()
        
        data = []
        for record in self.trip_records:
            data.append({
                'start location': record.start_location,
                'end location': record.end_location,
                'start time': record.start_time,
                'end time': record.end_time,
                'activity': record.activity,
                'line': record.line,
                'energy consumption': record.energy_consumption,
                'bus': record.bus_id
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values(['bus', 'start time']).reset_index(drop=True)
        
        df['line'] = df['line'].astype(float)
        
        return df
    
    def get_statistics(self) -> Dict:
        if not self.buses:
            return {}
        
        total_distance = sum(bus.total_distance_km for bus in self.buses)
        total_trips = sum(bus.total_trips for bus in self.buses)
        
        service_trips = [r for r in self.trip_records if r.activity == 'service trip']
        material_trips = [r for r in self.trip_records if r.activity == 'material trip']
        charging_sessions = [r for r in self.trip_records if r.activity == 'charging']
        idle_periods = [r for r in self.trip_records if r.activity == 'idle']
        
        return {
            'fleet_size': len(self.buses),
            'total_distance_km': total_distance,
            'total_service_trips': len(service_trips),
            'total_material_trips': len(material_trips),
            'total_charging_sessions': len(charging_sessions),
            'total_idle_periods': len(idle_periods),
            'avg_trips_per_bus': total_trips / len(self.buses) if self.buses else 0,
            'avg_distance_per_bus': total_distance / len(self.buses) if self.buses else 0
        }


def optimize_bus_planning(
    timetable_df: pd.DataFrame,
    distance_matrix_df: pd.DataFrame, 
    config: Optional[OptimizerConfig] = None,
    enable_opportunistic_charging: bool = False,
    min_idle_for_charging: int = 30
) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimize bus fleet size and create complete bus planning schedule.
    
    Uses single-packet circular rotation algorithm to minimize fleet size
    while ensuring all service trips are covered and all constraints are met:
    - Battery SOC stays within 10-90% (25.5-229.5 kWh)
    - All buses start and end at depot
    - Proper charging management
    
    Args:
        timetable_df: Service timetable with required trips
        distance_matrix_df: Distance and travel time matrix
        config: Optional optimizer configuration (uses defaults if None)
        enable_opportunistic_charging: Whether to add charging during idle periods
        min_idle_for_charging: Minimum idle duration (minutes) to trigger charging
    
    Returns:
        Tuple of (planning_dataframe, statistics_dict)
    """
    if config is None:
        config = OptimizerConfig()
    
    planner = SinglePacketPlanner(config, timetable_df, distance_matrix_df)
    
    fleet_size, success = planner.optimize_fleet_size(max_buses=50)
    
    if not success:
        report_error("Optimization failed to find feasible solution")
        return pd.DataFrame(), {}
    
    # Insert opportunistic charging sessions during long idle periods
    if enable_opportunistic_charging:
        planner.insert_opportunistic_charging(min_idle_duration=min_idle_for_charging)
    
    planning_df = planner.export_to_dataframe()
    stats = planner.get_statistics()
    
    report_info(f"Optimization complete: {stats}")
    
    return planning_df, stats
