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

    #Initiate different reliability levels and grease states
    a1Dict = {0.01:0.21, 0.02:0.33, 0.03:0.44, 0.04:0.53, 0.05:0.62, 0.1:1.0}
    greaseStateList = ['virgin', 'degraded']
    
    #Prepare linearized unreliability plot
    x = np.linspace(0.25, 100.0, 400)
    U = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    Ustar = [np.log(np.log(1.0/(1.0-uval))) for uval in U]
    
    probLin = [np.log(np.log(1.0/(1.0-uval))) for uval in a1Dict.keys()]
    
    plt.figure(figsize=(5.2,5.2))
    plt.grid(True)
    plt.xscale('log')
    plt.xticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
    plt.yticks(Ustar,U)
    plt.xlim(1,100)
    plt.ylim(min(Ustar),max(Ustar))
    plt.xlabel('Time (Years)', fontsize = 12)
    plt.ylabel('Unreliability', fontsize = 12)
    
    #Plot unreliability bounds obtained with base model
    df = pd.read_csv('base_model_unreliability_envelopes.csv')
    baseVirgin = df['virgin']
    baseDegraded = df['degraded']
    plt.plot(baseVirgin, probLin,'g-', label = 'Mild Turbine Virgin Grease Base Model')
    plt.plot(baseDegraded, probLin,'r-', label = 'Extreme Turbine Degraded Grease Base Model')
    
    for greaseState in greaseStateList:
        
        failedYears = []
        
        for prob, a1 in a1Dict.items():
            
            # Preliminaries
            myDtype = tf.float32
            C = 6000
            Pu = 750
            a = -10/3                  # Slope of linearized SN-Curve in log10-log10 space
            b = (10/3)*np.log10(C)+np.log10(1e6)+np.log10(a1)  # Interception of linearized SN-Curve in log10-log10 space
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
            
            # Load and manipulate required tables
            df = pd.read_csv('aSKF.csv')
            data_aSKF = np.transpose(np.asarray(np.transpose(df))[1:])
            if data_aSKF.shape[1] == 1:
                data_aSKF = np.repeat(data_aSKF,2,axis=1)
            data_aSKF = np.expand_dims(data_aSKF,0)
            data_aSKF = np.expand_dims(data_aSKF,-1)
            space_aSKF = np.asarray([np.asarray(df['xval']),np.asarray([float(i) for i in df.columns[1:]])])
            table_shape_aSKF = data_aSKF.shape
            bounds_aSKF = np.asarray([[np.min(space_aSKF[0]),np.min(space_aSKF[1])],[np.max(space_aSKF[0]),np.max(space_aSKF[1])]])
            
            df = pd.read_csv('kappa_'+greaseState+'.csv')
            data_kappa = np.transpose(np.asarray(np.transpose(df))[1:])
            if data_kappa.shape[1] == 1:
                data_kappa = np.repeat(data_kappa,2,axis=1)
            data_kappa = np.expand_dims(data_kappa,0)
            data_kappa = np.expand_dims(data_kappa,-1)
            space_kappa = np.asarray([np.asarray(df['btemp']),np.asarray([float(i) for i in df.columns[1:]])])
            table_shape_kappa = data_kappa.shape
            bounds_kappa = np.asarray([[np.min(space_kappa[0]),np.min(space_kappa[1])],[np.max(space_kappa[0]),np.max(space_kappa[1])]])
            
            df = pd.read_csv('etac_'+greaseState+'.csv')
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
            
            if greaseState == 'virgin':
                failedYears.append(np.where(result[0] > 1)[0][0]/(6*24*365))
            elif greaseState == 'degraded':
                failedYears.append(np.where(result[2] > 1)[0][0]/(6*24*365))
                
        if greaseState == 'virgin':
            plt.scatter(failedYears, probLin, color = 'green', label = 'Mild Turbine Virgin Grease PINN')
        elif greaseState == 'degraded':
            plt.scatter(failedYears, probLin, color = 'red', label = 'Extreme Turbine Degraded Grease PINN')
            
    plt.legend(bbox_to_anchor=(0.0, 1.0), loc=2, borderaxespad=0., prop={'size': 9})
    plt.show()