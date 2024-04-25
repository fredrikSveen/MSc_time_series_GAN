## Necessary Packages
import numpy as np
import pandas as pd
import os
import random

from timegan import timegan_from_pretrained
from utils import list_to_df

def df_from_file(filename):
    curr_dir = os.getcwd()
    filepath = os.path.join(curr_dir, 'data', filename)
    df = pd.read_csv(filepath)
    return df


def list_from_file(filename):
    seq_len = 24 #They use 24 in the original TimeGAN implementation
    curr_dir = os.getcwd()
    filepath = os.path.join(curr_dir, 'data', filename)
    ori_data = np.loadtxt(filepath, delimiter = ",", skiprows = 1)
    # Flip the data to make chronological data
    ori_data = ori_data[::-1]

    # Preprocess the dataset
    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
            
    # # Mix the datasets (to make it similar to i.i.d)
    # idx = np.random.permutation(len(temp_data))    
    # data = []
    # for i in range(len(temp_data)):
    #     data.append(temp_data[idx[i]])
    # return data


def synt_data_from_file(filename):
    curr_dir = os.getcwd()
    filepath = os.path.join(curr_dir, 'synt_data', filename)
    df = pd.read_csv(filepath)
    df.name = "Synthetic data"
    return df

################################################################################

# Generate data from a pretrained model

def generate_from_pretrained(model, model_filename, orig_data, reproduce=False, orig_data_filename=""):
    synt_data = pd.DataFrame()
    if model == "timegan":
        ori_data = list()
        try:
            ori_data = list_from_file(orig_data_filename) #Need to import the data again, because timeGan needs it in list form, not df.
            print("The original data is imported as a list")
        except IsADirectoryError:
            print("When using timeGAN, the filename of the original data needs to be specified in the variable orig_data_filename")
        
        ## Newtork parameters
        parameters = dict()

        parameters['module'] = 'gru' 
        parameters['hidden_dim'] = 24
        parameters['num_layer'] = 3
        parameters['iterations'] = 10000
        parameters['batch_size'] = 128

        print("Starting data generation with TimeGAN")
        synt_data_list = timegan_from_pretrained(model_filename, ori_data, parameters, reproduce)
        synt_data = list_to_df(synt_data_list)
        synt_data.name = "Synthetic data"
        print("Data generation complete")
    else:
        print("The specified model type is not supported.")

    return synt_data


################################################################################

# Generate Multivariate sinus time series

def generate_sine_wave(dim, n, seed = np.random.randint(0, 2000), deterministic = False):
    sine_data = []
    if deterministic:
        for i in range(dim):
            freq = 0.4
            phase = 0
            sine_data.append([np.sin(freq * j + phase) for j in range(n)])
    
    else:
        print(f'Random generation uses seed {seed}')
        random.seed(seed)

        for i in range(dim):
            freq = random.uniform(0.25, 0.75)
            phase = random.uniform(0, 0.1)
            sine_data.append([np.sin(freq * j + phase) for j in range(n)])
        
    sine_data = np.transpose(sine_data)
    return sine_data