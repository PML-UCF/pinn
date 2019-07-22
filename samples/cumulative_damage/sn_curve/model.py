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

""" Build physics-informed recursive neural network for SN-Curve sample
"""

import numpy as np
import tensorflow as tf

from pinn.layers import CumulativeDamageCell
from pinn.layers.physics import SNCurve
from pinn.layers.core import inputsSelection

def create_model(a, b, batch_input_shape, da0RNN, ndex, myDtype, return_sequences = False, unroll = False):

    batch_adjusted_shape = (batch_input_shape[0], batch_input_shape[1], batch_input_shape[2]+1)
    dLSelction = inputsSelection(batch_adjusted_shape, ndex)

    n = batch_input_shape[0]
    da_input_shape = (n,2)
    
    daLayer = SNCurve(input_shape = da_input_shape, dtype = myDtype)
    daLayer.build(input_shape = da_input_shape)
    daLayer.set_weights([np.asarray([a,b], dtype = daLayer.dtype)])
    daLayer.trainable = False

    PINN = tf.keras.Sequential()
    PINN.add(dLSelction)
    PINN.add(daLayer)
    "-------------------------------------------------------------------------"
    CDMCell = CumulativeDamageCell(model = PINN,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = da0RNN)

    CDMRNN = tf.keras.layers.RNN(cell = CDMCell,
                                       return_sequences = return_sequences,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll = unroll)
    "-------------------------------------------------------------------------"
    model = tf.keras.Sequential()
    model.add(CDMRNN)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])
    
    return model
