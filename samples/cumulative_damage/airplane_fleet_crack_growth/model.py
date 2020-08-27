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

""" Build physics-informed recursive neural network
"""

import numpy as np

from tensorflow.python.framework import tensor_shape

import tensorflow as tf

from pinn.layers import ParisLaw, CumulativeDamageCell, StressIntensityRange


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[None, 2]),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def create_model(dkLayer, C, m, batch_input_shape, a0RNN, myDtype, return_sequences=False, unroll=False):
    da_input_shape = dkLayer.output.shape[-1]
    daLayer = ParisLaw(input_shape=(da_input_shape,), dtype=myDtype)
    daLayer.build(input_shape=da_input_shape)
    daLayer.set_weights([np.asarray([C, m], dtype=daLayer.dtype)])
    daLayer.trainable = False

    PINNhybrid = tf.keras.Sequential()
    PINNhybrid.add(dkLayer)
    PINNhybrid.add(daLayer)

    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model=PINNhybrid,
                                         batch_input_shape=batch_input_shape,
                                         dtype=myDtype,
                                         initial_damage=a0RNN)

    CDMRNNhybrid = tf.keras.layers.RNN(cell=CDMCellHybrid,
                                       return_sequences=return_sequences,
                                       return_state=False,
                                       batch_input_shape=batch_input_shape,
                                       unroll=unroll)

    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])

    return model


def create_physics_model(F, C, m, batch_input_shape, a0RNN, myDtype, return_sequences=False, unroll=False):
    dk_input_shape = tensor_shape.TensorShape([2])

    dkLayer = StressIntensityRange(input_shape=dk_input_shape, dtype=myDtype)
    dkLayer.build(input_shape=dk_input_shape)
    dkLayer.set_weights([np.asarray([F], dtype=dkLayer.dtype)])
    dkLayer.trainable = False

    da_input_shape = tensor_shape.TensorShape([None, 1])
    daLayer = ParisLaw(input_shape=da_input_shape, dtype=myDtype)
    daLayer.build(input_shape=da_input_shape)
    daLayer.set_weights([np.asarray([C, m], dtype=daLayer.dtype)])
    daLayer.trainable = False

    PINNhybrid = tf.keras.Sequential()
    PINNhybrid.add(dkLayer)
    PINNhybrid.add(daLayer)

    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model=PINNhybrid,
                                         batch_input_shape=batch_input_shape,
                                         dtype=myDtype,
                                         initial_damage=a0RNN)

    CDMRNNhybrid = tf.keras.layers.RNN(cell=CDMCellHybrid,
                                       return_sequences=return_sequences,
                                       return_state=False,
                                       batch_input_shape=batch_input_shape,
                                       unroll=unroll)

    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])

    return model
