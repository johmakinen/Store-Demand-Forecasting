
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

from test_stationarity import test_stationarity
from errors import errors

color = sns.color_palette()
sns.set_style('darkgrid')

# Import data from csv
train = pd.read_csv('train.csv')

#Format datetime

train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

# Format original data again to start fresh:

train1 = train[train['store'] == 1]
train1 = train1[train1['item'] == 1]

train1['year'] = train1['date'].dt.year - 2012
train1['month'] = train1['date'].dt.month
train1['day'] = train1['date'].dt.dayofyear
train1['weekday'] = train1['date'].dt.weekday

# print(train1.head)

# Read the holiday file

holidays = pd.read_csv('usholidays.csv', header=None, names=['date', 'holiday'])

# Format datetime again

holidays['date'] = pd.to_datetime(holidays['date'], format='%Y/%m/%d')

# print(holidays.head(3))

# Merge holidays into the data:

train1 = train1.merge(holidays, how="left", on="date")

# Make another column for checking if a day is a holiday.
# 0 = not a holiday, 1 = holiday

train1['isHoliday'] = pd.notnull(train1['holiday']).astype(int)

# Do dummies to the data
train1 = pd.get_dummies(train1, prefix=['month','holiday','weekday'], columns=['month','holiday','weekday'])

# print(train1.head(4))
# Lets add a list of exogenous variables which seem to have some effect on the item demand. From previous plots we
# know that holidays, time of the week, and time of the year are important. Thus we'll make them all variables.

exo_variables = ['date','year', 'day', 'isHoliday',
       'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
       'holiday_Christmas Day', 'holiday_Columbus Day',
       'holiday_Independence Day', 'holiday_Labor Day',
       'holiday_Martin Luther King Jr. Day', 'holiday_Memorial Day',
       'holiday_New Year Day', 'holiday_Presidents Day (Washingtons Birthday)',
       'holiday_Thanksgiving Day', 'holiday_Veterans Day', 'weekday_0',
       'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
       'weekday_6']

# Get exogenous variable data from the train1 data with dummies.
ex_data = train1[exo_variables]

# Set index correctly
ex_data = ex_data.set_index('date')
# print(ex_data.head(4))
# Set train1 index back to normal
train1 = train1.set_index('date')

# We'll need to forecast the next 3 months. So we will be forecasting the lasst 3 months of the training data from the
# previous data and see the results.

# Period to forecast
starti = '2017-10-01'
endi = '2017-12-30'

# Create SARIMAX model with exogenous variables
model1 = sm.tsa.statespace.SARIMAX(endog = train1.sales[:starti],
                                        exog = ex_data[:starti],
                                        trend='n', order=(7,1,0), seasonal_order=(0,1,1,7)).fit()

# Create a forecast columns to train1 data for easier plotting.
endi_1 = '2017-12-31'
train1['forecast'] = model1.predict(start = pd.to_datetime(starti), end= pd.to_datetime(endi_1),
                                            exog = ex_data[starti:endi],
                                            dynamic= True)
# Plot the sales and the forecast.
train1[starti:endi][['sales', 'forecast']].plot(figsize=(10, 6))
plt.show()

errors(train1[starti:endi]['sales'],train1[starti:endi]['forecast'])

# Errors are still over 20% and the plot shows a great difference between the prediction and the real values.
#  TODO: We'll need to find a more suitable way to forecast the sales than SARIMAX-models.