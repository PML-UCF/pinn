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
    
    a1 = 1.0
    a = -10/3                  # Slope of linearized SN-Curve in log10-log10 space
    b = (10/3)*np.log10(6000)+np.log10(1e6)+np.log10(a1)  # Interception of linearized SN-Curve in log10-log10 space
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
    
    df = pd.read_csv('BearingTemp.csv', index_col = None)
    df = df.dropna()
    BTempFleet = np.transpose(np.asarray(df))
       
    inputArray = np.dstack((cycFleet, PFleetLog, BTempFleet))
    selectCycle = [1]
    selectLoad = [2]
    selectBTemp = [3]
    
    inputTensor = ops.convert_to_tensor(inputArray, dtype = myDtype)
    batch_input_shape = inputTensor.shape
    
    d0RNN = ops.convert_to_tensor(d0RNN * np.ones((inputArray.shape[0], 1)), dtype=myDtype)
    
    Pu = 750
    
    df = pd.read_csv('aSKF.csv')
    data_aSKF = np.transpose(np.asarray(np.transpose(df))[1:])
    if data_aSKF.shape[1] == 1:
        data_aSKF = np.repeat(data_aSKF,2,axis=1)
    data_aSKF = np.expand_dims(data_aSKF,0)
    data_aSKF = np.expand_dims(data_aSKF,-1)
    space_aSKF = np.asarray([np.asarray(df['xval']),np.asarray([float(i) for i in df.columns[1:]])])
    table_shape_aSKF = data_aSKF.shape
    bounds_aSKF = np.asarray([[np.min(space_aSKF[0]),np.min(space_aSKF[1])],[np.max(space_aSKF[0]),np.max(space_aSKF[1])]])
    
    df = pd.read_csv('kappa_degraded.csv')
    data_kappa = np.transpose(np.asarray(np.transpose(df))[1:])
    if data_kappa.shape[1] == 1:
        data_kappa = np.repeat(data_kappa,2,axis=1)
    data_kappa = np.expand_dims(data_kappa,0)
    data_kappa = np.expand_dims(data_kappa,-1)
    space_kappa = np.asarray([np.asarray(df['btemp']),np.asarray([float(i) for i in df.columns[1:]])])
    table_shape_kappa = data_kappa.shape
    bounds_kappa = np.asarray([[np.min(space_kappa[0]),np.min(space_kappa[1])],[np.max(space_kappa[0]),np.max(space_kappa[1])]])
    
    df = pd.read_csv('etac_degraded.csv')
    data_etac = np.transpose(np.asarray(np.transpose(df))[1:])
    if data_etac.shape[1] == 1:
        data_etac = np.repeat(data_etac,2,axis=1)
    data_etac = np.expand_dims(data_etac,0)
    data_etac = np.expand_dims(data_etac,-1)
    space_etac = np.asarray([np.asarray(df['kappa']),np.asarray([float(i) for i in df.columns[1:]])])
    table_shape_etac = data_etac.shape
    bounds_etac = np.asarray([[np.min(space_etac[0]),np.min(space_etac[1])],[np.max(space_etac[0]),np.max(space_etac[1])]])
    
    # PINN Model
    model = create_model(a, b, Pu, data_aSKF, bounds_aSKF, table_shape_aSKF, data_kappa, bounds_kappa, table_shape_kappa, data_etac, bounds_etac, table_shape_etac, d0RNN, batch_input_shape, selectCycle, selectLoad, selectBTemp, myDtype, return_sequences = True)
    
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
    
    print("L2 Fatigue Lives")
    print("Mild, Nominal, Agressive")
    print(np.where(result[0] > 1)[0][0]/(6*24*365), np.where(result[1] > 1)[0][0]/(6*24*365), np.where(result[2] > 1)[0][0]/(6*24*365))