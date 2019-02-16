
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from test_stationarity import test_stationarity
from errors import errors
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
# plt.show()

# The plot shows a slight upward trend and some seasonality.
# Lets see how the data behaves monthly

sns.lineplot(x='date', y='sales', data=train_1[:28])
# plt.show()
# Plot shows some pattern during the month.

sns.boxplot(x='weekday', y='sales',data=train_1)
# plt.show()

# The boxplot shows that the sales are higher on the weekends. This also shows that there are some outliers in the data.
# Lets see the additive decomposition of the data.

from statsmodels.tsa.seasonal import seasonal_decompose
add_decomp = seasonal_decompose(train_1['sales'], model='additive', freq=365)
fig = plt.figure()
fig = add_decomp.plot()
# plt.show()

# From the additive decomposition we can clearly see the upward trend and the seasonal part. The timeseries is not
# stationary.


# test_stationarity(train_1['sales'])

# Looks like the train_1 timeseries is not stationary, even though the p-value of D-F-test is under 5%.
# Lets try to stationarize the data.

Dtrain_1 = train_1.sales - train_1.sales.shift(1)  # Shift data
Dtrain_1 = Dtrain_1.dropna(inplace=False)  # Drop NaN values

test_stationarity(Dtrain_1, window=12)  #Test the stationarity again

# Now the data plots look like a stationary timeseries. Also the p-value of D-F-test is extremely small thus
# the timesseries is now stationary.

# Now we can take a look at the ACF and PACF plots to determine the parameters for our model.

# The original data
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_1.sales, lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_1.sales, lags=30, ax=ax2)
# plt.show()

# The differenced data
fig2 = plt.figure(figsize=(12, 8))
ax1 = fig2.add_subplot(211)
fig2 = sm.graphics.tsa.plot_acf(Dtrain_1, lags=30, ax=ax1)
ax2 = fig2.add_subplot(212)
fig2 = sm.graphics.tsa.plot_pacf(Dtrain_1, lags=30, ax=ax2)
# plt.show()

# We can see that there's a weekly seasonality as the plots how spikes every 7th lag.

# We know the p d and q values of the supposed SARIMA model. I is 1 as the one difference stationarised the timeseries.
# p = 6 as the autocorrelations tend to zero after lag 6.
# We'll consider the MA part to be nonexistent.

model = sm.tsa.statespace.SARIMAX(train_1.sales, trend='n', order=(6, 1, 0)).fit()
# print(model.summary())

# The summary for the model looks good. Now we'll look at the residuals of the model. They should be white noise.
from scipy import stats
from scipy.stats import normaltest

residuals = model.resid
print(normaltest(residuals))
# Normaltest has a very low p-value so the residuals are not normally distributed.

# Lets plot the residuals

fig = plt.figure(figsize=(10, 6))
ax0 = fig.add_subplot(111)

sns.distplot(residuals, fit=stats.norm, ax=ax0)

(mu, sigma) = stats.norm.fit(residuals)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')
# plt.show()

# The residuals seem to look like they come from the normal distribution. Lets use this model to predict the last
# 30 days and evaluate the prediction w.r.t. the real values.

start = 1730
end = 1826
train_1['forecast'] = model.predict(start = start, end= end, dynamic=True)
train_1[start:end][['sales', 'forecast']].plot(figsize=(12, 8))
# plt.show()

errors(train_1[1730:1825]['sales'], train_1[1730:1825]['forecast'])

# The prediction is not optimal. Lets add exogenous variables (Holidays of USA).
# We'll continue with that in prediction2.py file.

# CLOSE ALL PLOTS!
plt.close('all')
# !!!!!








