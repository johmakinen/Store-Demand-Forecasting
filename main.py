
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

color = sns.color_palette()
sns.set_style('darkgrid')

# Import data from csv
train = pd.read_csv('train.csv')

# Format datetime
train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

# Take a look at the data
# print(train.head(10))

# Lets look at the store No. 1 and item 1.
train_1 = train[train['store'] == 1]
train_1 = train_1[train['item'] == 1]


# Format datetimes correctly
train_1['year'] = train['date'].dt.year
train_1['month'] = train['date'].dt.month
train_1['day'] = train['date'].dt.dayofyear
train_1['weekday'] = train['date'].dt.weekday

print(train_1.head())

# Check a rough plot of the data, is there trend or seasonality?

sns.lineplot(x='date', y='sales', data=train_1)
plt.show()

# The plot shows a slight upward trend and some seasonality.
# Lets see how the data behaves monthly

sns.lineplot(x='date', y='sales', data=train_1[:28])
plt.show()
# Plot shows some pattern during the month.

sns.boxplot(x='weekday', y='sales',data=train_1)
plt.show()

# The boxplot shows that the sales are higher on the weekends. This also shows that there are some outliers in the data.
# Lets see the additive decomposition of the data.

from statsmodels.tsa.seasonal import seasonal_decompose
add_decomp = seasonal_decompose(train_1['sales'], model='additive', freq=365)
fig = plt.figure()
fig = add_decomp.plot()
plt.show()

# From the additive decomposition we can clearly see the upward trend and the seasonal part. The timeseries is not
# stationary.

# Define a function to test stationarity with Dickey-Fuller test and plots.
# The function is from https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries, window=12, cutoff=0.01):

    # Determine rolling statistics

    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=20)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    print(dfoutput)


test_stationarity(train_1['sales'])

# Looks like the train_1 timeseries is not stationary, even though the p-value of D-F-test is under 5%.
# Lets try to stationarize the data.

Dtrain_1 = train_1.sales - train_1.sales.shift(1)  # Shift data
Dtrain_1 = Dtrain_1.dropna(inplace=False)  # Drop NaN values


test_stationarity(Dtrain_1, window=12)  #Test the stationarity again

# Now the data plots look like a stationary timeseries. Also the p-value of D-F-test is extremely small thus
# the timesseries is now stationary.

