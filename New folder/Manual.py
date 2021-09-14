import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_excel("Estoque Sales.xlsx")

data = df["estoque"]
training, testing = data[:700], data[700:]
#plt.plot(data)
#plot_acf(data)
#plot_pacf(data)
pm.auto_arima(training, trace = True, stepwise= True)


