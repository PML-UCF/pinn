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
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

import sys
sys.path.append('../../../')

from pinn.layers import CumulativeDamageCell
from pinn.layers.physics import SNCurve, StressIntensityRange, WalkerModel
from pinn.layers.core import inputsSelection, SigmoidSelector

# Model
def create_model(a, b, F, beta, gamma, Co, m , alpha, ath, a0RNN, batch_input_shape, selectsn, selectdK, selectprop, selectsig, myDtype, return_sequences = False, unroll = False):
    
    batch_adjusted_shape = (batch_input_shape[2]+1,) #Adding state
    placeHolder = Input(shape=(batch_input_shape[2]+1,)) #Adding state
    
    filtersnLayer = inputsSelection(batch_adjusted_shape, selectsn)(placeHolder)
    
    filterdkLayer = inputsSelection(batch_adjusted_shape, selectdK)(placeHolder)
    
    filterdaLayer = inputsSelection(batch_adjusted_shape, selectprop)(placeHolder)
    
    filterssLayer = inputsSelection(batch_adjusted_shape, selectsig)(placeHolder)
    
    sn_input_shape = filtersnLayer.get_shape()
    
    snLayer = SNCurve(input_shape = sn_input_shape, dtype = myDtype)
    snLayer.build(input_shape = sn_input_shape)
    snLayer.set_weights([np.asarray([a,b], dtype = snLayer.dtype)])
    snLayer.trainable = False
    snLayer = snLayer(filtersnLayer)
    
    dk_input_shape = filterdkLayer.get_shape()
        
    dkLayer = StressIntensityRange(input_shape = dk_input_shape, dtype = myDtype)
    dkLayer.build(input_shape = dk_input_shape)
    dkLayer.set_weights([np.asarray([F], dtype = dkLayer.dtype)])
    dkLayer.trainable = False
    dkLayer = dkLayer(filterdkLayer)
    
    wmInput = Concatenate(axis = -1)([dkLayer, filterdaLayer])
    da_input_shape = wmInput.get_shape()
    
    wmLayer = WalkerModel(input_shape = da_input_shape, dtype = myDtype)
    wmLayer.build(input_shape = da_input_shape)
    wmLayer.set_weights([np.asarray([beta, gamma, Co, m], dtype = wmLayer.dtype)])
    wmLayer.trainable = False
    wmLayer = wmLayer(wmInput)
    
    ssaux = Concatenate(axis = -1)([filterssLayer, snLayer])
    ssInput = Concatenate(axis = -1)([ssaux, wmLayer])
    ss_input_shape = ssInput.get_shape()
    
    ssLayer = SigmoidSelector(input_shape = ss_input_shape, dtype = myDtype)
    ssLayer.build(input_shape = ss_input_shape)
    ssLayer.set_weights([np.asarray([alpha, ath], dtype = ssLayer.dtype)])
    ssLayer.trainable = False
    ssLayer = ssLayer(ssInput)

    functionalModel = Model(inputs=[placeHolder], outputs=[ssLayer])
    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model = functionalModel,
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
