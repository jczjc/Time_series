import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from math import exp
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score




raw_df = pd.read_excel("Raw_data_timeseries.xlsx")
raw_sales = raw_df["Volume"]
adjusted_df = pd.read_excel("log_diff_timeseries.xlsx")
adjusted = adjusted_df["Adjusted_Volume"]
train, test = adjusted[:60], adjusted[60:]
plot_acf(train) #MA(4)
plot_pacf(train, lags = 20) 



# Test 4 Models: AR, MA, ARMA

# Part1 AR:


def ar_train(lag: int, training):
    """
    """
    model = ARIMA(training, order= (lag,0,0))
    model_fit = model.fit()
    predicted = model_fit.predict(start = 0, end = len(training)-1, dynamic=False)
    rsquared = r2_score(training, predicted)   
    plt.plot(training, label = 'Training Data')
    plt.plot(predicted, label = 'Fit')
    plt.legend(loc="best")
    plt.title("AR training of lag {}, R^2 = {}".format(lag, rsquared))
    plt.show()

 
def ar_crossvalidate(lag, training, testing):
    """
    """
    model = ARIMA(training, order = (lag,0,0))
    model_fit = model.fit()  
    predictions = model_fit.predict(start=len(training), end=len(training)+len(testing)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(testing, predictions))
    plt.plot(test, label = "testing data")
    plt.plot(predictions, label = "predictions", color = "orange")   
    plt.legend(loc="best")
    plt.title("AR Cross Validate of lag {}, rmse = {}".format(lag, rmse))
    plt.show()
    
   
#ar_train(12, train)  
#ar_crossvalidate(12, train, test)

# Out of Sample forecast (1 year):
#ar_model = ARIMA(train, order= (12,0,0))
#ar_model_fit = ar_model.fit()
#start_index = len(adjusted)-1
#end_index  = start_index + 48
#forecast = ar_model_fit.predict(start =start_index, end = end_index)
#plt.plot(adjusted, label = 'True')
#plt.plot(forecast, label = 'Predicted')
#plt.legend(loc = 'best')
#plt.show()



# Part2 MA:


def ma_training(orders, training):
    """
    """
    model = ARIMA(training, order=(0,0,orders))
    model_fit = model.fit()
    predicted = model_fit.predict(start = 0, end = len(training)-1, dynamic = False)
    rsquared = r2_score(training, predicted)   
    plt.plot(training, label = 'Training Data')
    plt.plot(predicted, label = 'Fit')
    plt.legend(loc="best")
    plt.title("Moving Avg training of order {}, R^2 = {}".format(orders, rsquared))
    plt.show()    
    

def ma_crossvalidate(orders, training, testing):
    """
    """
    model = ARIMA(training, order = (0,0,orders))
    model_fit = model.fit()  
    predictions = model_fit.predict(start=len(training), end=len(training)+len(test)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(testing, predictions))
    plt.plot(test, label = "testing data")
    plt.plot(predictions, label = "predictions", color = "orange")   
    plt.legend(loc="best")
    plt.title("Moving Avg Cross Validate of order {}, rmse = {}".format(orders, rmse))
    plt.show()
    
#ma_training(4, train)
#ma_crossvalidate(4, train, test)

# Out of Sample forecast (48 weeks 2019):
#ma_model = ARIMA(train, order= (0,0,4))
#ma_model_fit = ma_model.fit()
#start_index = len(adjusted)-1
#end_index  = start_index + 48
#forecast = ma_model_fit.predict(start =start_index, end = end_index)
#plt.plot(adjusted, label = 'True')
#plt.plot(forecast, label = 'Predicted')
#plt.legend(loc = 'best')
#plt.show()



# part3 : ARMA


def ARMA_train(lag: int, orders:int, training):
    """
    """
    model = ARIMA(training, order= (lag,0,orders))
    model_fit = model.fit()
    predicted = model_fit.predict(start = 0, end = len(training)-1, dynamic=False)
    rsquared = r2_score(training, predicted)   
    plt.plot(training, label = 'Training Data')
    plt.plot(predicted, label = 'Fit')
    plt.legend(loc="best")
    plt.title("ARMA: AR({}), MA({}) training, R^2 = {}".format(lag, orders,rsquared))
    plt.show()
    
    
def ARMA_crossvalidate(lag, orders, training, testing):
    """
    """
    model = ARIMA(training, order = (lag,0,orders))
    model_fit = model.fit()  
    predictions = model_fit.predict(start=len(training), end=len(training)+len(testing)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(testing, predictions))
    plt.plot(test, label = "testing data")
    plt.plot(predictions, label = "predictions", color = "orange")   
    plt.legend(loc="best")
    plt.title("ARMA: AR({}), MA({}) cross validate, rmse = {}".format(lag, orders, rmse))
    plt.show()

#ARMA_train(12,4,train)
#ARMA_crossvalidate(12,4,train,test)

Out of Sample forecast (48 weeks 2019):
#arma_model = ARIMA(train, order= (12,0,12))
#arma_model_fit = arma_model.fit()
#start_index = len(adjusted)
#end_index  = start_index + 48
#forecast = arma_model_fit.predict(start =start_index, end = end_index)
#plt.plot(adjusted)
#plt.plot(forecast)
#plt.show()


    

#Part 4 ARIMA


def ARIMA_train(lag: int, integrated: int, orders:int, training):
    """
    """
    model = ARIMA(training, order= (lag,integrated,orders))
    model_fit = model.fit()
    predicted = model_fit.predict(start = 0, end = len(training)-1, dynamic=False)
    rsquared = r2_score(training, predicted)   
    plt.plot(training, label = 'Training Data')
    plt.plot(predicted, label = 'Fit')
    plt.legend(loc="best")
    plt.title("ARIMA: AR({}),I({}),MA({}) training, R^2 = {}".format(lag,integrated,orders,rsquared))
    plt.show()
    

def ARIMA_crossvalidate(lag, integrated, orders, training, testing):
    """
    """
    model = ARIMA(training, order = (lag,integrated,orders))
    model_fit = model.fit()  
    predictions = model_fit.predict(start=len(training), end=len(training)+len(testing)-1, dynamic=False)
    rmse = sqrt(mean_squared_error(testing, predictions))
    plt.plot(test, label = "testing data")
    plt.plot(predictions, label = "predictions", color = "orange")   
    plt.legend(loc="best")
    plt.title("ARIMA: AR({}),I({}),MA({}) cross validate, rmse = {}".format(lag,integrated,orders,rmse))
    plt.show()
    

#ARIMA_train(12,1,4,train)
#ARIMA_crossvalidate(12,1,4, train,test)

# Out of Sample forecast (48 weeks 2019):
#arima_model = ARIMA(adjusted, order= (18,2,8))
#arima_model_fit = arima_model.fit()
#start_index = len(adjusted)
#end_index  = start_index + 80
#forecast = arima_model_fit.predict(start =start_index, end = end_index)
#plt.plot(adjusted)
#plt.plot(forecast)
