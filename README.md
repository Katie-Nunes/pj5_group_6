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
The app runs on port 8501 using Streamlit's development server. The workflow is configured to start automatically.

Command: `streamlit run app.py`

## Configuration
Streamlit is configured via `.streamlit/config.toml`:
- Port: 8501
- Address: 0.0.0.0 (localhost)
- Available: bus.k-nunes.com
