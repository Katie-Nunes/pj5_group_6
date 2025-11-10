import pandas as pd

# Load the DataFrame
df = pd.read_excel("Excel Files/gantt_export (3).xlsx")

# Convert time columns to datetime
df['start time'] = pd.to_datetime(df['start time'])
df['end time'] = pd.to_datetime(df['end time'])

# Step 1: Calculate cumulative energy consumption per bus
df['cumulative energy'] = df.groupby('bus')['energy consumption'].cumsum()

# Step 2: Initialize the new bus column, and add an activity column for tracking charging activity
df['new bus'] = df['bus']  # Initially, new bus is same as the original bus ID
df['location'] = df['start location']  # Start by using the initial location column
df['activity'] = df['activity']  # Copy the original activity column

threshold = 255
garage = []  # List to track buses that are sent to the garage

# Iterate over the rows and update the bus line
for i in range(1, len(df)):
    current_bus_id = int(df.loc[i - 1, 'bus'])  # Get the current bus ID as integer

    # Step 1: Check if the cumulative energy exceeds the threshold of 255
    if df.loc[i, 'cumulative energy'] >= threshold:
        # Bus needs to be replaced (increment the bus ID) and sent to the garage
        print(f"Bus {current_bus_id} exceeds the threshold. Sending to garage.")

        # Add the current bus to the garage (location is updated to 'ehvgar' for the next row)
        df.loc[i, 'location'] = 'ehvgar'  # Set the current bus location to the garage
        df.loc[i, 'activity'] = 'charging'  # Change the activity to "charging"

        # Add the current bus to the garage list for later reuse
        garage.append(current_bus_id)

        # Increment the bus ID and assign a new bus for the next trip
        new_bus_id = current_bus_id + 1
        df.loc[i, 'new bus'] = str(new_bus_id)  # Assign the new bus ID
        print(f"Creating new Bus {new_bus_id}.")

        # Reset the cumulative energy for the new bus, since it's starting fresh
        df.loc[i, 'cumulative energy'] = df.loc[i, 'energy consumption']

    # Step 2: If the bus hasn't exceeded the threshold, continue with the same bus
    if df.loc[i, 'cumulative energy'] < threshold:
        # If the bus has not exceeded the threshold, it continues with the same bus ID
        df.loc[i, 'new bus'] = str(current_bus_id)  # Keep using the same bus
        df.loc[i, 'location'] = df.loc[i, 'start location']  # No change to location, continue on route
        df.loc[i, 'activity'] = df.loc[i, 'activity']  # Keep the same activity

    # If the current trip requires a new bus, check for buses in the garage
    if df.loc[i, 'cumulative energy'] >= threshold:
        # Check if there are any buses available in the garage
        if garage:
            # Get the first available bus from the garage
            new_bus_id = garage.pop(0)
            df.loc[i, 'new bus'] = str(new_bus_id)  # Assign the bus from the garage
            print(f"Using Bus {new_bus_id} from the garage.")

            # Reset the cumulative energy for the new bus
            df.loc[i, 'cumulative energy'] = df.loc[i, 'energy consumption']
            df.loc[i, 'location'] = df.loc[i, 'start location']  # Update location for the new bus
            df.loc[i, 'activity'] = 'active'  # Change the activity to "active"

# View the updated DataFrame
print(df[['bus', 'new bus', 'cumulative energy', 'energy consumption', 'location', 'activity']])

df.to_excel("CUCOO.xlsx")