import numpy as np


def errors(true_values, predicted_values):
    mape = np.mean(abs((true_values-predicted_values)/true_values))*100
    smape = np.mean((np.abs(predicted_values - true_values) * 200/ (np.abs(predicted_values) + np.abs(true_values))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f' % (mape, smape), "%")