# Store-Demand-Forecasting
The objective is to predict 3 months of item-level sales data at different store locations.  
We'll be focusing on one store and one item at first to find a suitable way of forecasting the demand.  
Data is from Kaggle "Store Item Demand Forecasting" -competition. https://www.kaggle.com/ 

Prediction 2 SMAPE is about 20% which is quite large. Need to find out a more suitable forecasting method than SARIMAX.  

# File descriptions
prediction1.py - First prediction without exogenous variables  (SARIMA-model)  
prediction2.py - Second prediction with exogenous variables (SARIMAX-model)  
usholidays.csv - Holidays of USA for exogenous variables  
train.csv - Training data  
test.csv - Test data (Note: the Public/Private split is time based)  
errors.py & test_stationarity.py are files for helper functions  


# Data fields

date - Date of the sale data.
store - Store ID  
item - Item ID  
sales - Number of items sold at a particular store on a particular date.  
