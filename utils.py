"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8, reproduce=False):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  if reproduce:
    seed = 1
    idx = np.random.default_rng(seed=seed).permutation(no)
  else:
    idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
  """Basic RNN Cell.
    
  Args:
    - module_name: gru, lstm, or lstmLN
    
  Returns:
    - rnn_cell: RNN Cell
  """
  assert module_name in ['gru','lstm','lstmLN']
  
  # GRU
  if (module_name == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module_name == 'lstm'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module_name == 'lstmLN'):
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


def random_generator (batch_size, z_dim, T_mb, max_seq_len, reproduce=False):
  """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
  Z_mb = list()
  s = 1 #random seed for data generation
  if reproduce:
    for i in range(batch_size):
      temp = np.zeros([max_seq_len, z_dim])
      temp_Z = np.random.default_rng(seed=s).uniform(0., 1, [T_mb[i], z_dim])
      temp[:T_mb[i],:] = temp_Z
      Z_mb.append(temp_Z)
      s += 1
  else:
    for i in range(batch_size):
      temp = np.zeros([max_seq_len, z_dim])
      temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
      temp[:T_mb[i],:] = temp_Z
      Z_mb.append(temp_Z)
  return Z_mb


def batch_generator(data, time, batch_size, reproduce=False):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  if reproduce:
    seed = 1
    idx = np.random.default_rng(seed=seed).permutation(no)
  else:
    idx = np.random.permutation(no)
  train_idx = idx[:batch_size]     
            
  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)
  
  return X_mb, T_mb

###########################################################################################
# Other helpful functions 

def list_to_df(list_):
    # The input list is 3-d because the data is separated in overlapping windows,
    # The list is firstly transformed to 2d
    list_2d = []
    for mat in list_:
        list_2d.append(mat[0])
    # Then the 2d list is transformed into a dataframe
    generated_df = pd.DataFrame(list_2d, columns=list(range(9)), dtype=float)
    return generated_df

def df_plot_separate(df, title=""):
  df.loc[:500].plot(subplots=True, layout=(3,3), figsize=(16, 10))

  if title == "":
    try:
      title = df.name
    except:
      title = "Dataframe with features plotted separately"
  plt.suptitle(title)
  plt.tight_layout()
  plt.show()

def plot_loc_right(df, title=None, subplots=False, figsize=(12,8)):
    axes = df.plot(title=title, subplots=subplots, figsize=figsize)
    axes = axes.flat  # .ravel() and .flatten() also work

    # extract the figure object to use figure level methods
    fig = axes[0].get_figure()

    # iterate through each axes to use axes level methods
    for ax in axes:
        
        ax.legend(loc='right', fontsize=14)
        
    plt.show()