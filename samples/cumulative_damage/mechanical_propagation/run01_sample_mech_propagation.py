# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:56:02 2019

@author: ar679403
"""

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

""" Walker model sample 
"""

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

import sys
sys.path.append('../../../')
from pinn.layers.physics import WalkerModel
# =============================================================================
# Functions
# =============================================================================
def create_model(F,alpha, gamma, Co, m , batch_input_shape, da0RNN, myDtype, return_sequences = False, unroll = False):
    dk_input_shape = batch_input_shape
    
    dkLayer = StressIntensityRange(input_shape = dk_input_shape, dtype = myDtype)
    dkLayer.build(input_shape = dk_input_shape)
    dkLayer.set_weights([np.asarray([F], dtype = dkLayer.dtype)])
    dkLayer.trainable = False
    
    wm_input_shape = tensor_shape.TensorShape([None, 2])
    wmLayer = WalkerModel(input_shape = wm_input_shape, dtype = myDtype)
    wmLayer.build(input_shape = wm_input_shape)
    wmLayer.set_weights([np.asarray([alpha,gamma,Co,m], dtype = wmLayer.dtype)])
    wmLayer.trainable = False
    
    PINNhybrid = tf.keras.Sequential()
    PINNhybrid.add(dkLayer)
    PINNhybrid.add(wmLayer)

    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model = PINNhybrid,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = a0RNN)
     
    CDMRNNhybrid = tf.keras.layers.RNN(cell = CDMCellHybrid,
                                       return_sequences = return_sequences,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll = unroll)

    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])       
    return model
#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = tf.float32  # defining type for the layer
    
    df = pd.read_csv('Walker_model_data.csv', index_col = None) # loading required data
    dK = df['dK'].values # stress intensity values for 10 different machines at a given instant t
    R = df['R'].values # stress ratio values for 10 different machines at a given instant t
    gamma = threshold(R) # Walker model coefficient
    
    
    
    input_array = np.asarray([dK,R])
    input_array = np.transpose(input_array)
    
    alpha,gamma = -1e8,.68 # Walker model customized sigmoid function parameters
    Co,m = 1.1323e-10,3.859 # Walker model coefficients (similar to Paris law) 
    #--------------------------------------------------------------------------
    danp = walker(dK,R,Co,m) # prediction of the genereic function
    
    batch_input_shape = (input_array.shape[-1],)
    
    model = create_model(alpha = alpha, gamma = gamma, Co = Co, m = m, batch_input_shape = batch_input_shape, myDtype = myDtype)
    results = model.predict_on_batch(input_array) # custumized layer prediction
    #--------------------------------------------------------------------------
    fig  = plt.figure(2)
    fig.clf()
    
    plt.plot(dK,danp,'ok', label = 'numpy')
    plt.plot(dK,results,'sm', label = 'tf Layer')
    
    
    plt.title('Walker model response')
    plt.xlabel('$\Delta$ K [MPa $m^{1/2}$]')
    plt.ylabel('$\Delta$ a [m]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')
    #--------------------------------------------------------------------------
  
    git che