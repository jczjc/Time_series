import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from numpy import log
#from math import sqrt
import statsmodels as sm
import pmdarima as pm
#from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#from statsmodels.tsa.arima.model import ARIMA
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error



df = pd.read_excel("Raw_data_timeseries.xlsx")
sales = df["Volume"]
train,test = sales[:60], sales[60:]
#decompose = seasonal_decompose(sales, model = "additive", period= 40)
#print(decompose.resid)
#decompose.plot()
#plt.show()


### ACF and PACF plots
#plot_acf(sales, lags = 40)
#plt.show()
#plot_pacf(sales, lags = 45)
#plt.show()


#model = auto_arima(sales[:60], start_p=1, start_q=1,
                           #max_p=8, max_q=8, m=4,
                           #start_P=0, seasonal=True,
                           #d=0, D=1, trace=True,
                           #error_action='ignore',  
                           #suppress_warnings=True,
                           #stepwise = False,
                           #random = False)





# Approach 1: Log and Differencing
# Note that by differencing we are sacrificisng the data for the first month of Jan 2017

def differencing(series, lag):
    """
    Return a list of difference of lag of the series inputted/
    """
    result = []
    for i in range(lag, len(series)):
        diff = series[i] - series[i-lag]
        result.append(diff)
    return result

#sales_log = log(sales)
#adjusted_monthly = differencing(sales_log, 4)
#adjusted_monthly = sales_log.diff(periods = 4)
#adjusted_monthly_ = adjusted_monthly.diff(periods = 1)
#plot_acf(adjusted_monthly_[5:], lags = 30)
#plt.show()
#plot_pacf(adjusted_monthly_[5:], lags = 30)
#plt.show()

fit = pm.arima.ARIMA(order=(2,0,2), seasonal_order=(2,1,2,4)).fit(y=sales[:60]) 
forecast = fit.predict()
plt.plot(sales[:60])
plt.plot(sales[60:])
plt.plot(forecast)
plt.show()



#new_df = pd.DataFrame(adjusted_monthly)
#new_df["Per Weekly"] = timestamp
#new_df.set_index("Per Weekly", inplace= True)
#new_df.to_excel("log_diff_timeseries.xlsx")

#plt.plot(adjusted_monthly_, color='orange', label= 'Log and Seasonally Differenced')
#plt.plot(sales, color = 'blue', label = 'Raw')
#plt.legend(loc='best')
#plt.show()

# Perform Adfuller test
#adfresult = adfuller(adjusted_monthly)
#print(adfresult)
# -6.055851488913725
# 1.245736295408003e-07
# {'1%': -3.530398990560757, '5%': -2.9050874099328317, '10%': -2.5900010121107266}

# Perform Kpss test
#kpsstest = kpss(adjusted_monthly)
#print(kpsstest)
# 0.12113890009959143
# 0.1
# {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}








    

