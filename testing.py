from check_innacuracies import check_for_innacuracies
import pandas as pd
import plotly.express as px


planning_df = pd.read_excel('Excel Files/Bus Planning.xlsx')
timetable_df = pd.read_excel('Excel Files/Timetable.xlsx')
distancematrix_df = pd.read_excel('Excel Files/DistanceMatrix.xlsx')

def make_gantt(dataframe):
    fig = px.timeline(dataframe, x_start="start_dt", x_end="finish_dt", y="bus", color="activity",
                      hover_data=["start location", "end location", "line", "energy consumption"],
                      title="Bus Planning – Daily Gantt")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Time", yaxis_title="", legend_title="Activity", font_size=13, title_font_size=22)
    return fig

expected_columns = ['bus', 'start time', 'end time', 'start location', 'end location', 'activity', 'energy consumption',
                    'line']
expected_dtypes = {'bus': 'object', 'start time': 'object', 'end time': 'object', 'start location': 'object',
                   'end location': 'object', 'activity': 'object', 'energy consumption': 'float64', 'line': 'object'}
df = check_for_innacuracies(planning_df, expected_columns, expected_dtypes, timetable_df, distancematrix_df)
print("tits")
fig = make_gantt(df)
fig.show()


