"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from utils import extract_time, rnn_cell, random_generator, batch_generator


def timegan_from_pretrained(modelname, ori_data, parameters, reproduce=False):
  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)        
  
  # Network Parameters
  hidden_dim   = parameters['hidden_dim'] 
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module'] 
  z_dim        = dim
  gamma        = 1

  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(modelname)
    saver.restore(sess,tf.train.latest_checkpoint('trained_models/./'))
    graph = tf.get_default_graph()

    #Paramters
    X = graph.get_tensor_by_name("myinput_x:0")
    Z = graph.get_tensor_by_name("myinput_z:0")
    T = graph.get_tensor_by_name("myinput_t:0")

    X_hat = graph.get_tensor_by_name("op_to_restore:0") # Remember to chenge this to myinput_xhat

    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len, reproduce=reproduce)
    generated_data_curr = list()
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
    generated_data = list()
    
    for i in range(no):
      temp = generated_data_curr[i,:ori_time[i],:]
      generated_data.append(temp)
          
    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val
    
    return generated_data