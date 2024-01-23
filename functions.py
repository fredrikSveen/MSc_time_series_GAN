import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.stats import entropy

###########################################################################
# Functions for plotting
def plot_sensor_data(name, df):
    x = df[name]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # plt.xticks(np.arange(min(x), max(x)+1, 500.0))
    plt.plot(x, label=name)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.title(f'Time series of sensor: {name}')
    plt.show()

def plot_all_nine_sensors(dataframe, title="Timeseries of 9 sensors"):
    plt.figure(figsize=(15, 9))
    plt.suptitle(title)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    rows = 3
    cols = 3
    n = rows * cols
    index = 1
    for name, df in dataframe.items():
        plt.subplot(rows, cols, index)
        index += 1
        plt.plot(df)
        plt.gcf().autofmt_xdate()
        plt.title(name)
    plt.show()

def plot_kl(KL, title="KL divergence"):
    # print(KL)
    # im = plt.imshow(KL, norm=colors.LogNorm())
    # plt.colorbar(im)

    sn.heatmap(KL, annot=True)
    plt.title(title)
    plt.show()

###########################################################################

# Functions for reshaping
def reshape_df(df):
    return df.resample('30S').mean().ffill() #Using ffill to replace Nan with their previous value. "30S" means resample to 1 sample per 30 seconds.

def reshape_dfs(dfs):
    reshape_dfs = {}

    for name, df in dfs.items():
        r_df = reshape_df(df).fillna(method='ffill')
        print(f'Original shape: {df.shape} Resampled shape: {r_df.shape}')
        reshape_dfs[name] = r_df

    print("Reshaping done")    
    return reshape_dfs

###########################################################################

#Functions for correlation

def correlate_dfs(standard_df, title = "Correlation between sensors"):
    all_dfs = pd.DataFrame()
    for name, df in standard_df.items():
        all_dfs[name] = df.reset_index(drop=True)

    corr = all_dfs.corr()
    corr.style.background_gradient()
    sn.heatmap(corr, annot=True)
    plt.title(title)
    plt.show()


###########################################################################
    
# Functions for standardization

def standardize_df(df):
    avg = df.mean().iloc[0]
    std = df.std().iloc[0]
    tmp_df = (df-avg)/std
    return tmp_df, avg, std

def standardize_dfs(dfs):
    standard_df = {}
    for name, df in dfs.items():
        tmp_df,avg,std = standardize_df(df)
        # print(f'{name}: Avg: {avg} Std: {std} Shape: {tmp_df.shape}')
        standard_df[name] = tmp_df

    return standard_df

###########################################################################

# Functiond for KL Divergence

def KL_with_params(m1, std1, m2, std2):
    return np.log(std2/std1)+(std1**2 + (m1-m2)**2)/(2*std2**2)-1/2

def kl_dfs(dfs):
    KL = []

    for name, df in dfs.items():
        avg1 = df.mean().iloc[0]
        std1 = df.std().iloc[0]
        row = []
        for name, df2 in dfs.items():
            avg2 = df2.mean().iloc[0]
            std2 = df2.std().iloc[0]
            kl_tmp = KL_with_params(avg1, std1, avg2, std2)
            row.append(kl_tmp)
        KL.append(row)
    return KL

def mv_kl_dfs(dfs):
    KL = []

    for df in dfs.values():
        df = df.dropna()
        avg1 = np.mean(df, axis=0)
        cov_mat1 = np.cov(df, rowvar=0)
        row = []
        for df2 in dfs.values():
            df2 = df2.dropna()
            avg2 = np.mean(df2, axis=0)
            cov_mat2 = np.cov(df2, rowvar=0)
            kl_tmp = kl_mvn(avg1, cov_mat1, avg2, cov_mat2)
            row.append(kl_tmp)
        KL.append(row)
    return KL
   

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 