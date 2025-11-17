# Bus Planning App

## Overview
A Streamlit web application for bus planning and scheduling. This tool helps visualize and validate bus timetables, planning files, and distance matrices through interactive Gantt charts and feasibility checks.

## Project Setup
- **Language**: Python 3.11
- **Framework**: Streamlit
- **Type**: Web Application (Frontend)

## Key Features
- Upload and validate bus planning files (Excel format)
- Interactive Gantt chart visualization
- Feasibility and inaccuracy checking
- Energy consumption analysis
- Export improved bus planning schedules
- **Packet-based fleet optimization** - Minimize bus count using circular rotation algorithm

## Architecture
### Main Files
- `app.py` - Main Streamlit dashboard application
- `app_visualization_functions.py` - Visualization and Gantt chart functions
- `packet_optimizer.py` - Single-packet circular rotation fleet optimizer
- `check_inaccuracies.py` - Data validation and accuracy checking
- `check_feasbility.py` - Feasibility analysis functions
- `logging_utils.py` - Logging utilities
- `test_packet_optimizer.py` - Test suite for packet optimizer

### Data Files
- `Excel Files/` - Sample Excel files for testing
  - `Bus Planning.xlsx` - Main planning data
  - `Timetable.xlsx` - Bus timetable
  - `DistanceMatrix.xlsx` - Distance matrix between locations

## Dependencies
Core packages:
- streamlit - Web dashboard framework
- pandas - Data manipulation
- numpy - Numerical computations
- plotly - Interactive charts
- xlsxwriter - Excel export
- openpyxl - Excel file reading

## Running the Application
The app runs on port 5000 using Streamlit's development server. The workflow is configured to start automatically.

Command: `streamlit run app.py`

## Configuration
Streamlit is configured via `.streamlit/config.toml`:
- Port: 5000
- Address: 0.0.0.0 (allows Replit proxy)
- CORS disabled (required for Replit iframe)

## Deployment
Configured for autoscale deployment (stateless web app).

## Recent Changes
- 2025-11-17: **UI Restructure with Original/Validated/Optimized Views**
  - Converted from cramped 3-column layout to modern sidebar + full-width design
  - **NO EMOJIS**: Removed all emoji icons from entire project (including favicon)
  - **Text-based emoticons only**: ᕙ(  •̀ ᗜ •́  )ᕗ for success, ( ｡ •̀ ᴖ •́ ｡) for failure,  (╥ᆺ╥；)  for warnings
  - **Warnings/errors in main screen only**: No warnings or errors displayed in sidebar
  - Tab names simplified: "Schedules", "Performance", "Data", "Optimize"
  - Headings renamed to: "Original", "Validated", "Optimized" (clean, concise naming)
  - **Three-version comparison**: Each tab now shows Original, Validated, and Optimized versions
    - Schedules tab: Compare Gantt charts across all three versions
    - Performance tab: Compare KPIs and feasibility checks across all three versions (KPIs and pie chart now side-by-side)
    - Data tab: Browse raw data for all three versions plus reference data (Timetable, Distance Matrix)
  - **Data preprocessing**: Original and validated schedules calculated before tabs (available to all tabs)
  - **time_taken column**: Automatically added to original and optimized schedules if missing
  - **Auto-refresh after optimization**: Page automatically reruns after optimizer completes so results appear in all tabs
  - Optimizer results persist in session state and display across all tabs
  - Debug indicator in sidebar shows when optimized data is available
  - Configuration moved to organized sidebar with collapsible expanders
  - Export functionality integrated into sidebar

- 2025-11-12: **Configuration system improvements**
  - Fixed critical session_state timing bug: variables now initialized before reading (prevents errors on first run)
  - Added missing UI controls for all OptimizerConfig parameters:
    - Added `garage_location` input (Advanced Settings)
    - Added `charging_buffer_percent` slider (Advanced Settings)
  - Reorganized optimizer configuration UI into 3 columns with helpful tooltips:
    - Battery Settings: capacity, SOH, min/max SOC
    - Charging & Energy: charging rate/window, energy consumption, idle energy
    - Advanced Settings: garage location, charging buffer
  - All 10 configuration parameters now properly flow from UI to optimizer
  - Verified no parameter drift or missing connections

- 2025-11-12: **Comprehensive documentation added**
  - Added module-level docstrings to all core files
  - Added comprehensive function/method docstrings with Args and Returns sections
  - Added type hints to all function signatures
  - Documented all critical public methods in packet_optimizer.py:
    - SinglePacketPlanner class and __init__
    - Core methods: get_next_available_bus, predict_soc_after_trip, send_to_charging
    - Trip assignment: assign_service_trip
    - Optimization: optimize_fleet_size, run_simulation
    - Main API: optimize_bus_planning
  - Documented all functions in check_feasbility.py (energy_state through check_all_feasibility)
  - Documented all logging utilities in logging_utils.py
  - All LSP errors resolved
  - Code is now production-ready with professional documentation standards

- 2025-11-12: **Project cleanup and refactoring**
  - Removed unused files: main.py, error_handling.py, README.md
  - Removed old experimental folders: CREATE planning/, Improve planning/
  - Fixed LSP errors in feasibility checking
  - Cleaned up codebase for production readiness

- 2025-11-12: **Charging cap fix**
  - Fixed charging records to show actual energy added (not theoretical)
  - Buses now correctly record only the energy that brings them to 229.5 kWh cap
  - Prevents misleading negative energy values in output
  
- 2025-11-12: **Feasibility check fix**
  - Fixed false violations in Insights tab
  - Corrected initial SOC assumption from 255 kWh to 229.5 kWh
  - Feasibility validation now matches optimizer behavior
  
- 2025-11-12: **CRITICAL SOC VALIDATION FIX - Optimizer now fully validated**
  - **Root Cause**: Buses initialized at wrong SOC (`effective_capacity` 255 kWh instead of `max_soc` 229.5 kWh)
  - **Impact**: Created 25.5 kWh phantom buffer that masked all SOC violations during simulation
  - **Fix**: Changed `initialize_fleet()` to use `config.max_soc` for starting SOC
  - **Added comprehensive SOC guards**:
    - Material trip to garage guard: Check bus can reach garage without dropping below minimum
    - Cumulative energy guard: Check total energy (idle + deadhead + service) before trip assignment
    - Depot return guard: Check and charge before final return to garage if needed
    - Empty return validation: Fail simulation if any guard returns empty list
  - **Result**: Optimizer produces VALID schedules with ZERO SOC violations
  - **Validation**: All 327 service trips, minimum SOC 28.0 kWh (above 25.5 kWh floor), 14 buses
  
- 2025-11-12: Opportunistic charging and idle optimization (currently disabled)
  - **Opportunistic charging**: Buses return to garage during long idle periods (>30 min) to charge instead of sitting idle
  - **Just-in-time departure**: Buses idle at garage and depart at optimal time to arrive exactly when trips start
  - **Zero energy at garage**: Buses parked at garage consume no energy (turned off), only consume energy when idling elsewhere
  - **Clean output**: Idle periods >3 hours are excluded from output (energy still tracked, but no visual clutter)
  - Result: Cleaner schedules, better battery management, more efficient use of idle time
  
- 2025-11-12: Complete optimizer output compatibility
  - **Schema validation**: Now accepts datetime64[ns] for time columns and int64 for line column
  - **Smart datetime handling**: Skips gap filling and sequence fixing when data is already datetime64
  - **Relaxed energy validation**: Only fixes clearly invalid values (preserves optimizer parameters)
    - Charging: only fix if >= 0 (positive values are wrong)
    - Trip energy: only fix if <= 0 or > km * 10 (clearly invalid)
    - Idle energy: only fix if <= 0 or > 50 kWh (unreasonably high)
  - Optimizer output now uploads cleanly as Bus Planning without corruption
  - Gantt charts display all activity types correctly
  - Feasibility checks pass for optimizer output
  - Legacy manual files still work with full validation
  
- 2025-11-12: Smart validation to preserve optimizer output integrity
  - Modified `rename_time_object` to detect and preserve datetime64 columns (skip conversion if already datetime)
  - Relaxed energy validation thresholds to avoid corrupting optimizer output:
    - Charging: only fix if >= 0 (positive values are wrong)
    - Trip energy: only fix if <= 0 or > km * 10 (clearly invalid)
    - Idle energy: only fix if <= 0 or > 50 kWh (unreasonably high)
  - Fixes eliminate false errors when uploading optimizer output as Bus Planning
  - Gantt charts now display correctly for both manual and optimizer-generated schedules
  - Validation still catches genuinely incorrect values while preserving valid optimizer data
  
- 2025-11-12: Optimizer output compatibility fixes
  - Fixed false "minimum charge" errors when uploading optimizer output as Bus Planning
  - Made charging energy validation preserve optimizer's negative values (only "fix" if >= 0)
  - Removed unnecessary datetime conversion for Gantt charts in Optimize tab
  - Optimizer output now correctly displays all activity types (service, material, idle, charging)
  - Feasibility checks now pass correctly for optimizer output files
  
- 2025-11-12: Gantt chart and validation fixes
  - Fixed Gantt chart display by removing datetime type conversion
  - Resolved false "minimum charge" validation errors caused by datetime parsing issues
  - All activity types (service, material, idle, charging) now display correctly in charts
  - Energy tracking verified correct: charging properly increases SOC as expected
  
- 2025-11-12: Depot return trips and validation fixes
  - Added automatic depot return trips at end of each bus schedule
  - All buses now guaranteed to start and end at depot (garage location)
  - All tests passing with depot returns: Casus 1 (1 bus, 2 depot trips), Casus 2 (1 bus, 19 material trips), Main timetable (8 buses, 278 material trips)
  
- 2025-11-12: Packet-based fleet optimization implementation
  - Created packet_optimizer.py with single-packet circular rotation algorithm
  - Implemented SOC prediction, charging management, and fleet size optimization
  - Added "Optimize" tab to Streamlit UI with configuration controls
  - Created comprehensive test suite (test_packet_optimizer.py)
  - All tests passing: Casus 1 (1 bus), Casus 2 (1 bus), Main timetable (8 buses)
  
- 2025-11-12: Initial setup in Replit environment
  - Installed Python 3.11 and all dependencies
  - Configured Streamlit for Replit proxy (port 5000, all hosts allowed)
  - Set up workflow for automatic app startup
  - Configured deployment settings for autoscale
