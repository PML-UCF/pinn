# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:04:16 2019

@author: ar679403
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.python.framework import ops
# =============================================================================
# Function
# =============================================================================

def inputsSelection(inputs, ndex):
    input_mask = np.zeros(inputs.get_shape(), dtype = int)
    input_mask[:,ndex] = 1
# =============================================================================
#     input_mask = ops.convert_to_tensor(input_mask, dtype = tf.int32)
# =============================================================================
        
    dL = Dense(inputs.get_shape()[-1], activation = None, input_shape = inputs.shape, 
               use_bias = False)
    dL.build(input_shape = inputs.shape)
    dL.set_weights(input_mask)
    dL.trainable = False
    return dL
# =============================================================================
# Main
# =============================================================================
np.random.seed(123)
myDtype = tf.float32

input_array = np.random.random((10,5))
input_array = ops.convert_to_tensor(input_array, dtype = myDtype)
ndex = np.asarray([0,2,4])

dLSample = inputsSelection(input_array, ndex)