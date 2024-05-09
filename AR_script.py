# Import the library 
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
# Importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datautil.relativedelta import relativedelta
import sys
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 

sys.stdout.flush()

data = np.loadtxt(f'sine_data/sine_123_1000_9.csv', delimiter = ",",skiprows = 1)
# data = data[0]
data = pd.DataFrame(data)
# data = data[1]
print(type(data))

def get_forecast_group(data, n_periods, seasonal):
    # Initialize empty lists to store forecast data
    data_fc = []
    data_lower = []
    data_upper = []
    data_aic = []
    data_fitted = []
    
    # Iterate over columns in data
    for group in data.columns:
        # Fit an ARIMA model using the auto_arima function
        data_actual = data[group]
        model = pm.auto_arima(data_actual, 
                              start_p=0, start_q=0,
                              max_p=12, max_q=12, # maximum p and q
                              test='adf',         # use adftest to find optimal 'd'
                              seasonal=seasonal,  # TRUE if seasonal series
                              m=12,               # frequency of series
                              d=None,             # let model determine 'd'
                              D=None,             # let model determine 'D'
                              trace=False,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)
        
        # Generate forecast and confidence intervals for n_periods into the future
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = pd.date_range(pd.to_datetime(data_actual.index[-1])  + relativedelta(months = +1), periods = n_periods, freq = 'MS')
        
        # Append forecast data to lists
        data_fc.append(fc)
        data_lower.append(confint[:, 0])
        data_upper.append(confint[:, 1])
        data_aic.append(model.aic())
        data_fitted.append(model.fittedvalues())

        # Create dataframes for forecast, lower bound, and upper bound
        df_fc = pd.DataFrame(index = index_of_fc)
        df_lower = pd.DataFrame(index = index_of_fc)
        df_upper = pd.DataFrame(index = index_of_fc)
        df_aic = pd.DataFrame()
        df_fitted = pd.DataFrame(index = data_actual.index)

    # Populate dataframes with forecast data
    i = 0
    for group in data.columns:
        df_fc[group] = data_fc[i][:]
        df_lower[group] = data_lower[i][:]
        df_upper[group] = data_upper[i][:]
        df_aic[group] = data_aic[i]
        df_fitted[group] = data_fitted[i][:]
        i = i + 1
    
    return df_fc, df_lower, df_upper, df_aic, df_fitted

def get_combined_data(df_actual, df_forecast):
    # Assign input data to separate variables
    data_actual = df_actual
    data_forecast = df_forecast
    
    # Add a 'desc' column to indicate whether the data is actual or forecast
    data_actual['desc'] = 'Actual'
    data_forecast['desc'] = 'Forecast'
    
    # Combine actual and forecast data into a single DataFrame and reset the index
    df_act_fc = pd.concat([data_actual, data_forecast]).reset_index()
    
    # Rename the index column to 'month'
    df_act_fc = df_act_fc.rename(columns={'index': 'month'})

    # Return the combined DataFrame
    return df_act_fc

def get_plot_fc(df_act_fc, df_lower, df_upper, df_fitted, nrow, ncol, figsize_x, figsize_y, category_field_values,  title, ylabel):
    # Set the years and months locators and formatter
    years = mdates.YearLocator()    # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    # Melt the data for plotting
    df_melt = df_act_fc.melt(id_vars = ['month', 'desc'])
    df_melt_fitted = df_fitted.reset_index().melt(id_vars = ['month'])

    # Create subplots and set the title
    fig, axs = plt.subplots(nrow, ncol, figsize = (figsize_x,figsize_y))
    fig.suptitle(title, size = 20, y = 0.90)

    i = 0
    j = 0
    for cat in category_field_values:
        # Filter data for the current category
        df_plot = df_melt[df_melt['variable'] == cat]
        df_lower_plot = df_lower[cat]
        df_upper_plot = df_upper[cat]
        df_plot_fitted = df_melt_fitted[df_melt_fitted['variable'] == cat]

        # Plot the actual and forecasted data
        sns.lineplot(ax = axs[j,i], data = df_plot, x = 'month', y = 'value', hue = 'desc', marker = 'o')
        # Plot the fitted data with dashed lines
        sns.lineplot(ax = axs[j,i], data = df_plot_fitted, x = 'month', y = 'value', dashes=True, alpha = 0.5)
        # Set the x-label, y-label, and fill between the lower and upper bounds of the forecast
        axs[j, i].set_xlabel(cat, size = 15)
        axs[j, i].set_ylabel(ylabel, size = 15)
        axs[j,i].fill_between(df_lower_plot.index, 
                      df_lower_plot, 
                      df_upper_plot, 
                      color='k', alpha=.15)
        # Set the legend and y-limits
        axs[j,i].legend(loc = 'upper left')
        axs[j,i].set_ylim([df_plot['value'].min()-1000, df_plot['value'].max()+1000])

        # Set the x-axis tickers and format
        axs[j,i].xaxis.set_major_locator(years)
        axs[j,i].xaxis.set_major_formatter(years_fmt)
        axs[j,i].xaxis.set_minor_locator(months)

        i = i + 1 
        if i >= ncol:
            j = j + 1
            i = 0
    plt.savefig(f'AR_figs/test.png', bbox_inches='tight')
    plt.show()


df_fc, df_lower, df_upper, df_aic, df_fitted = get_forecast_group(data = data, 
                                                                  n_periods = 24, 
                                                                  seasonal = True)

df_act_fc = get_combined_data(df_actual = data, df_forecast = df_fc)

get_plot_fc(df_act_fc, 
            df_lower, 
            df_upper, 
            df_fitted,
            nrow = 5, ncol = 2, 
            figsize_x = 25, figsize_y = 25,
            category_field_values = df_act_fc.drop(['month', 'desc'], axis = 1).columns, 
            title = 'Total Bottle Sold on Top 10 Counties',
            ylabel = 'Bottles')