import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Creating Avg Monthly Sales Column
df = pd.read_excel("Raw_data_timeseries.xlsx")
sales = df["LineTrueGrams"]
sales_list = sales.to_list()
month_avg = []
i = 0
while i <= len(sales_list)-4:
    month_avg.append((sales_list[i] + sales_list[i+1] + sales_list[i+2] + sales_list[i+3])/4)
    i += 4
    
# Creating Month Column   
date = df["Date"]
date_list = date.to_list()
date_list_str = []
for timestamp in date_list:
    date_list_str.append(datetime.strftime(timestamp, '%Y-%m-%d'))
month = []
i = 0
while i <= len(date_list_str) -4:
    month.append(date_list_str[i][:7])
    i += 4 
    
# Creating new dataframe
new_df = pd.DataFrame(month, columns= ['Month'])
new_df['Average Sales'] = month_avg
new_df.set_index("Month", inplace = True)

# Write to New csv
new_df.to_excel('Monthly_sales_timeseries.xlsx')


    
