import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

color = sns.color_palette()
sns.set_style('darkgrid')

# Import data from csv
train = pd.read_csv('train.csv')

# Format datetime
train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

# Take a look at the data
print(train.head(10))



