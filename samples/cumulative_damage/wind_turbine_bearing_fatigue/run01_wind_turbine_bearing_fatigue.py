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

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.python.framework import ops

from model import create_model

if __name__ == "__main__":

    # Preliminaries
    myDtype = tf.float32
    
    a = -10/3                  # Slope of linearized SN-Curve in log10-log10 space
    b = (10/3)*np.log10(6000)+np.log10(1e6)  # Interception of linearized SN-Curve in log10-log10 space
    d0RNN = 0.0
    
    # Inputs
    df = pd.read_csv('Cycles.csv', index_col = None)
    df = df.dropna()
    cycFleet = np.transpose(np.asarray(df))
    
    df = pd.read_csv('DynamicLoad.csv', index_col = None)
    df = df.dropna()
    PFleet = np.transpose(np.asarray(df))
    PFleetLog = np.log10(PFleet)
    nFleet, n10min = PFleet.shape
       
    inputArray = np.dstack((cycFleet, PFleetLog))
    selectCycle = [1]
    selectLoad = [2]
    batch_input_shape = inputArray.shape
    inputTensor = ops.convert_to_tensor(inputArray, dtype = myDtype)
    
    d0RNN = ops.convert_to_tensor(d0RNN * np.ones((inputArray.shape[0], 1)), dtype=myDtype)
    
    # PINN Model
    model = create_model(a, b, d0RNN, batch_input_shape, selectCycle, selectLoad, myDtype, return_sequences = True)
    
    result = model.predict(inputArray)
    
    # Base Model
    def deltaDamage(C,P,n):
        L = 1e6*(C/P)**(10/3)
        delDmg = n/L
        return delDmg
    
    C = 6000
    dmgCum = [[],[],[]]
    dmg = np.zeros(nFleet)
    for t in range(n10min):
        dmg[0] +=  deltaDamage(C,PFleet[0][t],cycFleet[0][t])
        dmgCum[0].append(dmg[0])
        dmg[1] +=  deltaDamage(C,PFleet[1][t],cycFleet[1][t])
        dmgCum[1].append(dmg[1])
        dmg[2] +=  deltaDamage(C,PFleet[2][t],cycFleet[2][t])
        dmgCum[2].append(dmg[2])

    plt.plot(np.transpose(np.repeat(np.array([range(n10min)]),3,axis =0)),np.transpose(dmgCum),'-') 
    plt.plot(np.transpose(np.repeat(np.array([range(n10min)]),3,axis =0)),np.transpose(result[:,:,0]),'--')
    plt.legend(bbox_to_anchor=(0.0, 1.0), loc=2, borderaxespad=0.,labels = ('BaseMild','BaseNom','BaseAgro','PINNMild','PINNNom','PINNAgro'))
    plt.show()