
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

from prediction1 import errors
from prediction1 import test_stationarity

color = sns.color_palette()
sns.set_style('darkgrid')

# Import data from csv
train = pd.read_csv('train.csv')


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

print(holidays.head(3))

# Merge holidays into the data:

train1 = train1.merge(holidays,how="left", on="date")






