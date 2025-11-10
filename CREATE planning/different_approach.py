import pandas as pd
from datetime import datetime, date, timedelta

df = pd.read_excel("Excel Files/gantt_export (3).xlsx")

# Go through every row, check if row is idle row, at bus-station, and timetaken is greater than 24 mins, if so, replace that row
# with travel to garage charge timetaken-8 mins and back, if at airport, same as above but needs 55 mins at least

# concatenate bus lines. for every bus check if there is a service trip which leaves from a location at the same time as
# one is either charging or idling.
# Check if there is a service line whose start time and end time is between another one's and whose start and
# end location is also the same

df['start time'] = pd.to_datetime(df['start time'])
df['end time'] = pd.to_datetime(df['end time'])
df.loc[df['end time'] < df['start time'], 'end time'] += timedelta(days=1)

def recalc():
    df['time_taken'] = df['end time'] - df['start time']
    df['cumulative energy'] = df.groupby('bus')['energy consumption'].cumsum()

for row, idx in df:
    if df.at(idx, "time_taken") >= 24 and df.(idx, "activity") == "idle":
        # replace with garage trip
        pass
