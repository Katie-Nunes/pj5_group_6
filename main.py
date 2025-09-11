import pandas as pd

df = pd.read_excel('Excel Files/Bus Planning.xlsx')
df_tocomp = ['start location', 'end location', 'start time', 'end time', 'activity', 'line', 'energy consumption', 'bus']
timetable = pd.read_excel('Excel Files/Timetable.xlsx')
print(df.dtypes.to_dict())
def main (df):
    check_heading(df)
    check_against_datatype(df)

def check_heading (df):
    get_df_headings = df.columns.tolist()
    if get_df_headings == df_tocomp:
        print("True")
    else:
        print("Headings are not named the same, please fix")

def check_against_datatype (df):
    print(df.info())

def check_againt_possible_locations ():
    pass

def check_for_innacuracies ():
    pass

#def check_feasability ():
#    pass
