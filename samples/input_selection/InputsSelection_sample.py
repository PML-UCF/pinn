# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:04:16 2019

@author: ar679403
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
# =============================================================================
# Function
# =============================================================================
def inputsSelection(inputs, ndex):
    input_mask = np.zeros([inputs.shape[-1], len(ndex)])
    for i in range(inputs.shape[-1]):
        for v in range(len(ndex)):
            if i == v:
                input_mask[i,v] = 1
        
    dL = Dense(len(ndex), activation = None, input_shape = inputs.shape, 
               use_bias = False)
    dL.build(input_shape = inputs.shape)
    dL.set_weights([input_mask])
    dL.trainable = False
    return dL


def create_model(input_array, ndex):
    dLSelction = inputsSelection(input_array, ndex)
    model = tf.keras.Sequential()
    model.add(dLSelction)
    return model

# =============================================================================
# Main
# =============================================================================
np.random.seed(123)

input_array = np.random.random((10,5))
input_shape = input_array.shape
ndex = np.asarray([0,2,4])

test_model = create_model(input_array, ndex)
out = test_model.predict(input_array.reshape((1,10,5)))
print(out)