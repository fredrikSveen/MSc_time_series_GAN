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

data = np.loadtxt(f'sine_data/sine_123_1000_9.csv', delimiter = ",",skiprows = 1)
# data = data[0]
data = pd.DataFrame(data)
data = data[8]
print(type(data))

# Fit auto_arima function to AirPassengers dataset 
stepwise_fit = auto_arima(data, start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary() 