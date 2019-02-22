# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:28:16 2019

@author: ar679403
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

import sys
sys.path.append('../../')
from pinn.layers import CumulativeDamageCell
# =============================================================================
# Layers
# =============================================================================
class SN(Layer):
    """ SN curve:
        `output = a*inputs+b
        where:
            * `a`,`b` parametric constants,        
            * inputs stress.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SN, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape = [2],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        #inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        output = 1/10**(self.kernel[0]*inputs[:,1]+self.kernel[1])
        if(output.shape[0].value is not None):
            output = tf.reshape(output, (tensor_shape.TensorShape((output.shape[0],1))))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 
# =============================================================================
# NN framework
# =============================================================================
def create_model(a, b, batch_input_shape, da0RNN, myDtype, return_sequences = False, unroll = False):
    n = batch_input_shape[0]
    da_input_shape = (n,2)
    daLayer = SN(input_shape = da_input_shape, dtype = myDtype)
    daLayer.build(input_shape = da_input_shape)
    daLayer.set_weights([np.asarray([a,b], dtype = daLayer.dtype)])
    daLayer.trainable = False
    
    PINN = tf.keras.Sequential()
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
# =============================================================================
# Main
# =============================================================================
myDtype = tf.float32
P = np.linspace(240,6000,50)
a = -10/3
b = 13.372
N = a*np.log10(P)+b
d = 1/10**N
do = 0
cycles = 10

S = np.repeat(np.log10(P),cycles)
S = np.reshape(S,(np.shape(P)[0],cycles))

da = np.repeat(d,cycles)
da = np.reshape(da,(np.shape(P)[0],cycles))
da = np.sum(da,axis = 1)

Sobs = S[:, :, np.newaxis]
Sobs = ops.convert_to_tensor(Sobs, dtype = myDtype)
    
batch_input_shape = Sobs.shape

daTarget = da[:, np.newaxis]
daTarget = ops.convert_to_tensor(daTarget, dtype=myDtype)

da0RNN = ops.convert_to_tensor(do * np.ones((Sobs.shape[0], 1)), dtype=myDtype)
#dkLayer.trainable = True
    
model = create_model(a = a, b = b, batch_input_shape = batch_input_shape, da0RNN = da0RNN, myDtype = myDtype)

model.fit(Sobs, daTarget ,epochs=5, steps_per_epoch=5)

results = model.predict(Sobs, verbose=0, steps=1)
print(results)
"-------------------------------------------------------------------------"
ifig = 0
ifig = ifig + 1
fig  = plt.figure(ifig)
fig.clf()

plt.plot(1/da, P, '-', label = 'SN model')
plt.plot(1/(results), P, 'r--', label = 'PINN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('cycles')
plt.ylabel('load')
plt.grid(which = 'both')
plt.legend(loc=0, facecolor = 'w')           

