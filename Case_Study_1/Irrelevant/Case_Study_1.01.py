import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def columns_to_list(dataframe, column:str):
    """
    Return the desired column of the dataframe in a python list
    """
    return dataframe[column].tolist()

def string_to_datetime(column):
    """
    Return a list of datetime objects given a list of strings
    """
    new = []
    for i in column:
        new.append(datetime.strptime(i, '%Y-%m-%d'))
    return new 

    
df = pd.read_excel("Table_cs1.xlsx")
date_sales = df[['LineTrueGrams', 'DateKey']].loc[0:79]
sales= columns_to_list(date_sales, 'LineTrueGrams')
date_key_str = columns_to_list(date_sales, 'DateKey')
date_key_dt = string_to_datetime(date_key_str)
plt.plot_date(date_key_dt, sales, linestyle = 'solid', color = 'orange')
plt.show()


season_sales = df[['LineTrueGrams', 'seasonality']].loc[0:79]



