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
    
    df = pd.read_csv('Walker_init_Prop_data.csv', index_col = None) # loading required data
    Seq = df['Seq'].values
    dS = df['dS'].values # loads history for a given machine 
    R = df['R'].values # stress ratio values for a given machine
    cr = df['a'].values # crack length values for a given machine
    
    nFleet, nCycles = 3, len(cr) 
    
    Seq_fleet = np.repeat(Seq,nFleet) # simulating n identical machines
    Seq_fleet = np.reshape(Seq_fleet,(nCycles,nFleet))
    Seq_fleet = Seq_fleet.transpose()
    dS_fleet = np.repeat(dS,nFleet) # simulating n identical machines
    dS_fleet = np.reshape(dS_fleet,(nCycles,nFleet))
    dS_fleet = dS_fleet.transpose()
    R_fleet = np.repeat(R,nFleet) # simulating n identical machines
    R_fleet = np.reshape(R_fleet,(nCycles,nFleet))
    R_fleet = R_fleet.transpose()
    
    # RNN inputs
    input_array = np.dstack((Seq_fleet, dS_fleet))
    input_array = np.dstack((input_array, R_fleet))
    inputTensor = ops.convert_to_tensor(input_array, dtype = myDtype)
    
    a0RNN = np.round(cr[0],4) # initial crack length
    a0RNN = ops.convert_to_tensor(a0RNN * np.ones((input_array.shape[0], 1)), dtype=myDtype)
    
    # model parameters
    a,b = -3.73,13.48261 # Sn curve coefficients 
    F = 2.8 # stress intensity factor
    beta,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    Co,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law)
    alpha,ath = 1e6,.5e-3 # sigmoid selector parameters
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    
    selectsn = [1]
    selectdK = [0,2]
    selectprop = [3]
    selectsig = [0]
    
    model = create_model(a, b, F, beta, gamma, Co, m , alpha, ath, a0RNN, batch_input_shape, selectsn, selectdK, selectprop, selectsig, myDtype, return_sequences = True)
    results = model.predict_on_batch(input_array) # custumized layer prediction
    #--------------------------------------------------------------------------
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(1e3*cr,':k', label = 'data')
    plt.plot(1e3*results[0,:,0],':', label = 'PINN#1')
    plt.plot(1e3*results[1,:,0],'--', label = 'PINN#2')
    plt.plot(1e3*results[-1,:,0],'-', label = 'PINN#3')
    
    
    plt.title('Crack Init. and Prop.')
    plt.xlabel('Cycles')
    plt.ylabel(' crack length [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')