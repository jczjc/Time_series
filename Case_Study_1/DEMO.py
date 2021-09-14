import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
#import pmdarima as pm
from DateTime import DateTime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error




df = pd.read_excel("Raw_data_timeseries.xlsx")
#string= df["Per Weekly"].strftime("%Y-%m-%d")
#df.set_index("Per Weekly", inplace = True)
#index=df.index.strftime("%Y-%m-%d")
#df.set_index(index, inplace = True)
sales = df["Volume"]
train,test = sales[:65], sales[65:]
#x = np.arange(len(sales))
#plt.plot(df)


#ACF&PACF
#plot_acf(sales, lags = 30)
#plt.show()
#plot_pacf(sales, lags = 35)
#plt.show()

#Could Have seasonally differenced and then select parameters manually 
#Use autoarima for algorithmic search 

#parameters = pm.auto_arima(train,seasonal = False, m=4, trace = True, D=1,stepwise= True)
model = ARIMA(train, order=(3,0,0), seasonal_order=(0,1,1,4))
results = model.fit()
#results_ = results.predict
#print(results.summary())
testing = results.predict(start= 65, end= 79)
#rms = mean_squared_error(test, testing, squared=False)
plt.plot(train, label='Training Data')
plt.plot(testing, label= 'Testing Fit')
plt.plot(test, label='Testing Data')
plt.legend(loc='best')
plt.show()



