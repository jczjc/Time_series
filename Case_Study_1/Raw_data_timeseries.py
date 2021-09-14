import csv
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
    
df = pd.read_excel("Raw_data_only_table.xlsx")
date_sales = df[['LineTrueGrams', 'DateKey']].loc[0:79]
date_key_str = columns_to_list(date_sales, 'DateKey')
date_key_dt = string_to_datetime(date_key_str)
date_sales.drop(columns = ['DateKey'], inplace= True)
date_sales['Date'] = date_key_dt
date_sales.set_index('Date', inplace = True)
date_sales.to_csv("Raw_data_timeseries.csv")






