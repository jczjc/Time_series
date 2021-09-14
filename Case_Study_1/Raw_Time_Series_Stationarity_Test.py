import pandas as pd
from statsmodels.tsa.stattools import adfuller

df = pd.read_excel("Raw_data_timeseries.xlsx")

# Part 1. Summary Stats Check

# a) Split into 2
sales = df['Volume']
split = len(sales)//2
x1, x2 = sales[0:split], sales[split:]
mean1, mean2 = x1.mean(), x2.mean()
var1, var2 = x1.var(), x2.var()
#print("Mean: {} and {}".format(mean1, mean2))
#print("Variance: {} and {}".format(var1, var2))
#Mean: 94334.1 and 120869.95
#Variance: 2567329022.6051283 and 3155549478.7666664

# b) Local vs Global
local = sales[15:45]
mean3, mean4 = local.mean(), sales.mean()
var3, var4 = local.var(), sales.var()
#print("Mean: {} and {}".format(mean3, mean4))
#print("Variance: {} and {}".format(var3, var4))
#Mean: 95955.5 and 107602.025
#Variance: 2975848254.1896553 and 3003484661.493038

# c) Locals Comparison
splitby5 = (sales[:16], sales[16:32], sales[32:48], sales[48:64], sales[64:80])
i = 1
for split in splitby5:
    #print("Split{}: mean = {} and Variance = {}".format(i, split.mean(), split.var()))
    i += 1
#Split1: mean = 97726.25 and Variance = 3190076968.0666666
#Split2: mean = 88776.5625 and Variance = 2376637070.6625
#Split3: mean = 102494.1875 and Variance = 2906950814.829167
#Split4: mean = 117502.1875 and Variance = 2877152460.0291667
#Split5: mean = 131510.9375 and Variance = 3243355729.1291666


# Part 2. Augmented Dickey-Fuller Test
result = adfuller(sales)
adf_statistic = result [0]
p_value = result[1]
#print(adf_statistic)
#print(p_value)
for key in result[4]:
    #print("{}: {}".format(key, result[4][key]))
#adf = -0.8737166796160966
#p_value = 0.7965046395987576
#Criticial Values:
# 1%: -3.530398990560757
# 5%: -2.9050874099328317
# 10%: -2.5900010121107266


# Conclusion: The time-series of weekly sales volume is non-stationary.