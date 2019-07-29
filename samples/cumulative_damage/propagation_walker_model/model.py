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

from pinn.layers import inputsSelection, CumulativeDamageCell
from pinn.layers import StressIntensityRange, WalkerModel
# Model
def create_model(F, alpha, gamma, C0, m , d0RNN, batch_input_shape, input_array, selectdK, selectprop, myDtype, return_sequences = False, unroll = False):
    
    batch_adjusted_shape = (batch_input_shape[2]+1,) #Adding state
    placeHolder = Input(shape=(batch_input_shape[2]+1,)) #Adding state
    
    filterdkLayer = inputsSelection(batch_adjusted_shape, selectdK)(placeHolder)
    
    filterdaLayer = inputsSelection(batch_adjusted_shape, selectprop)(placeHolder)
    
    dk_input_shape = filterdkLayer.get_shape()
        
    dkLayer = StressIntensityRange(input_shape = dk_input_shape, dtype = myDtype)
    dkLayer.build(input_shape = dk_input_shape)
    dkLayer.set_weights([np.asarray([F], dtype = dkLayer.dtype)])
    dkLayer.trainable = False
    dkLayer = dkLayer(filterdkLayer)
    
    wmInput = Concatenate(axis = -1)([dkLayer, filterdaLayer])
    wm_input_shape = wmInput.get_shape()
    
    wmLayer = WalkerModel(input_shape = wm_input_shape, dtype = myDtype)
    wmLayer.build(input_shape = wm_input_shape)
    wmLayer.set_weights([np.asarray([alpha, gamma, C0, m], dtype = wmLayer.dtype)])
    wmLayer.trainable = False
    wmLayer = wmLayer(wmInput)

    functionalModel = Model(inputs=[placeHolder], outputs=[wmLayer])
    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model = functionalModel,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = d0RNN)
     
    CDMRNNhybrid = tf.keras.layers.RNN(cell = CDMCellHybrid,
                                       return_sequences = return_sequences,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll = unroll)
    
    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])
    return model
