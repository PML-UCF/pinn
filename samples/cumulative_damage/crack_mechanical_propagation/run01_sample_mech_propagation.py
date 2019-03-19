# ______          _           _     _ _ _     _   _      
# | ___ \        | |         | |   (_) (_)   | | (_)     
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___ 
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__ 
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _                  
# |  \/  |         | |               (_)                 
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___         
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|        
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \        
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/        
#  _           _                     _                   
# | |         | |                   | |                  
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _ 
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/ 
#														  
# MIT License
# 
# Copyright (c) 2019 Probabilistic Mechanics Laboratory
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
""" Mechanical propagation sample 
"""
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from model import create_model
#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = tf.float32  # defining type for the layer
    
    df = pd.read_csv('Propagation_loads_n_crack_length_data.csv', index_col = None) # loading required data
    dS = df['dS'].values # loads history for a given machine 
    R = df['R'].values # stress ratio values for a given machine 
    
    n = len(dS) # number of loads
    
    dS_fleet = np.repeat(dS,10) # simulating ten identical machines
    dS_fleet = np.reshape(dS_fleet,(10,n))
    
    R_fleet = np.repeat(R,10) 
    R_fleet = np.reshape(R_fleet,(10,n))
    
    # RNN inputs
    input_array = np.dstack((dS_fleet, R_fleet))
    inputTensor = ops.convert_to_tensor(input_array, dtype = myDtype)
    
    a0RNN = 2e-3 # initial crack length
    a0RNN = ops.convert_to_tensor(a0RNN * np.ones((input_array.shape[0], 1)), dtype=myDtype)
    
    # model parameters
    F = 2.8 # stress intensity factor
    alpha,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    Co,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law) 
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    
    selectdK = [0,1]
    selectprop = [2]
    
    model = create_model(F, alpha, gamma, Co, m , a0RNN, batch_input_shape, input_array, selectdK, selectprop, myDtype, return_sequences = True)
    results = model.predict_on_batch(input_array) # custumized layer prediction
    #--------------------------------------------------------------------------
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(dS,1e3*df['a'].values,'ok', label = 'numpy')
    plt.plot(dS,1e3*results,'sm', label = 'PINN')
    
    
    plt.title('Mech. Propagation')
    plt.xlabel('$\Delta$ S [MPa]')
    plt.ylabel('$\Delta$ a [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')