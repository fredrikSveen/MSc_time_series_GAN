# Import the library 
from pmdarima import auto_arima
# Importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 

import os
directory = os.path.join(os.curdir, "sine_data")
file = 'sine_123_1000_9.csv'

data = np.loadtxt(os.path.join(directory, file), delimiter = ",",skiprows = 1)
dim = file.split('_')[3][0]
print(f'dim is {dim}')

data = pd.DataFrame(data)

if dim == '1':
    data = data[1]
    best_model = auto_arima(data, start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 

else:
    # Fit auto_arima function to dataset
    generated_sines = []
    for feature in data.columns:
        feat = data[feature]
        best_model = auto_arima(feat, start_p = 1, start_q = 1, 
                                max_p = 3, max_q = 3, 
                                start_P = 0, seasonal = True, 
                                d = None, D = 1, trace = True, 
                                error_action ='ignore',   # we don't want to know if an order does not work 
                                suppress_warnings = True,  # we don't want convergence warnings 
                                stepwise = True)           # set to stepwise
        forecast = pd.DataFrame(best_model.predict(n_periods = 5))
        print(forecast)
        
  