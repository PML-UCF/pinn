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
import matplotlib.pyplot as plt

from model import create_model
#--------------------------------------------------------------------------
if __name__ == "__main__":
    
    myDtype = 'float32'
    
    dfa = pd.read_csv('Crack_length.csv', index_col = None, dtype = myDtype) # crack length data
    cr = dfa.values[:,1:4] # crack length values for all machines
    cr = cr.transpose() # setting axis as [# of machines, # of cycles]
    idex = np.where(cr[1,:]>.5e-3)[0][0] # index to split data into initiation - propagation stages
    cr = cr[:,idex:-1]
    dfdS = pd.read_csv('Delta_load.csv', index_col = None, dtype = myDtype) # Load data
    dS = dfdS.values[:,1:4] # loads history for all machines 
    dS = dS.transpose() 
    dS = dS[:,idex:-1]
    dfR = pd.read_csv('Stress_ratio.csv', index_col = None, dtype = myDtype) # Stress ratio data
    R = dfR.values[:,1:4] # stress ratio values for all machines
    R = R.transpose()
    R = R[:,idex:-1]
        
    nFleet, nCycles  = np.shape(cr) 
    
    # RNN inputs
    input_array = np.dstack((dS, R))
    
    a0RNN = np.zeros((input_array.shape[0], 1), dtype = myDtype) 
    a0RNN[0] = cr[0,0] # initial crack length asset #1
    a0RNN[1] = cr[1,0] # initial crack length asset #2
    a0RNN[-1] = cr[-1,0] # initial crack length asset #3
    
    # model parameters
    F = 2.8 # stress intensity factor
    alpha,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    C0,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law) 
    #--------------------------------------------------------------------------
    batch_input_shape = input_array.shape
    
    selectdK = [0,1]
    selectprop = [2]
    
    model = create_model(F, alpha, gamma, C0, m , a0RNN, batch_input_shape, input_array, selectdK, selectprop, myDtype, return_sequences = True)
    results = model.predict_on_batch(input_array) # custumized layer prediction
    #--------------------------------------------------------------------------
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(1e3*cr[0,:],':k', label = 'asset #1')
    plt.plot(1e3*cr[1,:],'--m', label = 'asset #2')
    plt.plot(1e3*cr[-1,:],'-g', label = 'asset #3')
    plt.plot(1e3*results[0,:,0],':', label = 'PINN #1')
    plt.plot(1e3*results[1,:,0],'--', label = 'PINN #2')
    plt.plot(1e3*results[-1,:,0],'-', label = 'PINN #3')
             
    plt.title('Mech. Propagation')
    plt.xlabel('Cycles')
    plt.ylabel('$\Delta$ a [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')
    