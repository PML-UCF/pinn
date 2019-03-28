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

""" Input selection sample case
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../')
from pinn.layers.core import SigmoidSelector
# =============================================================================
# functions
# =============================================================================
def create_model(alpha, ath, batch_input_shape, myDtype):
    thLayer = SigmoidSelector(input_shape = batch_input_shape, dtype = myDtype)
    thLayer.build(input_shape = batch_input_shape)
    thLayer.set_weights([np.asarray([alpha,ath], dtype = thLayer.dtype)])
    thLayer.trainable = False
    
    model = tf.keras.Sequential()
    model.add(thLayer)
        
    return model

def threshold(dai,dap,a,ath): # implementation of the layer in matrix form for comparison purposes
    alpha = 1e6
    m = 1/(1+np.exp(-alpha*(a-ath)))
    da = m*dap+(1-m)*dai
    return da
# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    myDtype = tf.float32  # defining type for the layer
    
    df = pd.read_csv('Crack_info_50_machines.csv', index_col = None) # loading crack length data
    input_array = np.asarray([df['a'],df['dai'],df['dap']])
    input_array = np.transpose(input_array)
    
    a = df['a'].values # crack lengths for 50 different machines at a given instant t.
    dai = df['dai'].values # crack length increment in the initiation stage
    dap = df['dap'].values # crack length increment in the propagation stage
    
    ath = .5e-3  # crack length that defines the transition from initiation to propagation stage. 
    alpha = 1e6  # constant required by the customized sigmoid function in the layeer
    
      
    danp = threshold(dai,dap,a,ath) # prediction of the genereic function
    
    batch_input_shape = (input_array.shape[-1],)
    
    model = create_model(alpha = alpha, ath = ath, batch_input_shape = batch_input_shape, 
                         myDtype = myDtype)
    results = model.predict_on_batch(input_array) # custumized layer prediction
    #--------------------------------------------------------------------------
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(a*1e3,dai*1e3,':r', label = 'init.')
    plt.plot(a*1e3,dap*1e3,':b', label = 'prop.')
    plt.plot(a*1e3,danp*1e3,'ok', label = 'numpy')
    plt.plot(a*1e3,results*1e3,'sm', label = 'tf Layer')
    
    
    plt.title('Sigmoid Selector function response')
    plt.xlabel('crack length [mm]')
    plt.ylabel('$\Delta$ a [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')
    #--------------------------------------------------------------------------
    

