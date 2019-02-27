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
def inputsSelection(inputs_shape, ndex):
    input_mask = np.zeros(inputs_shape[-1], dtype = int)
    input_mask[ndex] = 1
    input_mask = np.diag(input_mask)
        
    dL = Dense(inputs_shape[-1], activation = None, input_shape = inputs_shape, 
               use_bias = False)
    dL.build(input_shape = inputs_shape)
    dL.set_weights([input_mask])
    dL.trainable = False
    return dL

def create_model(ndex, batch_input_shape):
    dLSelction = inputsSelection(batch_input_shape, ndex)
    model = tf.keras.Sequential()
    model.add(dLSelction)
    return model
# =============================================================================
# Main
# =============================================================================
np.random.seed(123)
myDtype = tf.float32

input_array = np.random.random((10,5))
input_array = ops.convert_to_tensor(input_array, dtype = myDtype)
ndex = np.asarray([0,2,4])

batch_input_shape = input_array.shape

test_model = create_model(ndex, batch_input_shape)
test_model.predict_on_batch(input_array)