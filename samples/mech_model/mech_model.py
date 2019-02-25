# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 19:07:36 2019

@author: ar679403
"""

""" Build physics-informed recursive neural network
"""

import numpy as np

import tensorflow as tf
from physics_layers import EquivalentStress, InitiationSN, StressIntensityRange, ParisLaw

import sys
sys.path.append('../../')

from pinn.layers import CumulativeDamageCell


def create_model(Seq_exp, cs1, cs2, cs3, ao, F, C, m, batch_input_shape, a0RNN, myDtype, return_sequences = False, unroll = False):
    
    SeqLayer = EquivalentStress(input_shape = batch_input_shape, dtype = myDtype)
    SeqLayer.build(input_shape = batch_input_shape)
    SeqLayer.set_weights([np.asarray([Seq_exp], dtype = SeqLayer.dtype)])
    SeqLayer.trainable = False
    
    da_init_input_shape = SeqLayer.get_output_shape_at(-1)
    da_init_Layer = InitiationSN(input_shape = da_init_input_shape, dtype = myDtype)
    da_init_Layer.build(input_shape = da_init_input_shape)
    da_init_Layer.set_weights([np.asarray([cs1, cs2, cs3, ao], dtype=da_init_Layer.dtype)])
    da_init_Layer.trainable = False
    
# =============================================================================
#     dkLayer = StressIntensityRange(input_shape = batch_input_shape, dtype = myDtype)
#     dkLayer.build(input_shape = batch_input_shape)
#     dkLayer.set_weights([np.asarray([F], dtype = dkLayer.dtype)])
#     dkLayer.trainable = False
#     
#     da_prop_input_shape = dkLayer.get_output_shape_at(-1)
#     da_prop_Layer = ParisLaw(input_shape = da_prop_input_shape, dtype = myDtype)
#     da_prop_Layer.build(input_shape = da_prop_input_shape)
#     da_prop_Layer.set_weights([np.asarray([C, m], dtype=da_prop_Layer.dtype)])
#     da_prop_Layer.trainable = False
# =============================================================================
            
    PINNinit = tf.keras.Sequential()
    PINNinit.add(SeqLayer)
    PINNinit.add(da_init_Layer)

    "-------------------------------------------------------------------------"
    CDMCellinit = CumulativeDamageCell(model = PINNinit,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = a0RNN)
     
    CDMRNNinit = tf.keras.layers.RNN(cell = CDMCellinit,
                                       return_sequences = return_sequences,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll = unroll)
    "-------------------------------------------------------------------------"

# =============================================================================
#     PINNprop = tf.keras.Sequential()
#     PINNprop.add(dkLayer)
#     PINNprop.add(da_prop_Layer)
# 
#     "-------------------------------------------------------------------------"
#     CDMCellprop = CumulativeDamageCell(model = PINNprop,
#                                        batch_input_shape = batch_input_shape,
#                                        dtype = myDtype,
#                                        initial_damage = a0RNN)
#      
#     CDMRNNprop = tf.keras.layers.RNN(cell = CDMCellprop,
#                                        return_sequences = return_sequences,
#                                        return_state = False,
#                                        batch_input_shape = batch_input_shape,
#                                        unroll = unroll)
# =============================================================================
    "-------------------------------------------------------------------------"
    model = tf.keras.Sequential()
# =============================================================================
#     model.add([CDMRNNinit,CDMRNNprop])
# =============================================================================
    model.add(CDMRNNinit)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])
    
    return model
