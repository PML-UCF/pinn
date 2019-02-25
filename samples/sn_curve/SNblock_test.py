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
cycles = 1000
machines = 50
P = np.linspace(240,6000,50)

import random
maclist = []
for m in range(machines):
    loadhist = []
    for c in range(cycles):
        loadhist.append(P[m]*(random.random()+0.5))
    maclist.append(loadhist)
    
macarray = np.asarray(maclist)

a = -10/3
b = 13.372
N = a*np.log10(macarray)+b
d = 1/10**N
do = 0
dlast = [np.sum(d[i]) for i in range(machines)]
dlast = np.asarray(dlast)

dhistall = []
for mac in d:
    dmgcum = 0
    dhist = []
    for cyc in mac:
        dmgcum += cyc
        dhist.append(dmgcum)
    dhistall.append(dhist)
dhistall = np.asarray(dhistall)

Sobs = macarray[:, :, np.newaxis]
Sobs = np.log10(Sobs)
Sobs = ops.convert_to_tensor(Sobs, dtype = myDtype)

batch_input_shape = Sobs.shape

# Training 
daTarget = dhistall[:,-1]
daTarget = daTarget[:, np.newaxis] 
daTarget = ops.convert_to_tensor(daTarget, dtype=myDtype)

da0RNN = ops.convert_to_tensor(do * np.ones((Sobs.shape[0], 1)), dtype=myDtype)
    
pre_model = create_model(a = a, b = b, batch_input_shape = batch_input_shape, 
                         da0RNN = da0RNN, myDtype = myDtype)
history = pre_model.fit(Sobs, daTarget ,epochs=5, steps_per_epoch=5)

pre_model.save_weights('model_weights.h5')

model = create_model(a = a, b = b, batch_input_shape = batch_input_shape, 
                     da0RNN = da0RNN, myDtype = myDtype, return_sequences = True)
model.load_weights('model_weights.h5')

results = model.predict_on_batch(Sobs)[:,:,0]
"-------------------------------------------------------------------------"
ifig = 0
ifig = ifig + 1
fig  = plt.figure(ifig)
fig.clf()

plt.plot(range(cycles), dhistall[0],'b-', label = 'Machine 1 SN model')
plt.plot(range(cycles), results[0],'r--', label = 'Machine 1 SN PINN')
plt.plot(range(cycles), dhistall[1],'k-', label = 'Machine 2 SN model')
plt.plot(range(cycles), results[1],'y--', label = 'Machine 2 SN PINN')
plt.plot(range(cycles), dhistall[2],'c-', label = 'Machine 3 SN model')
plt.plot(range(cycles), results[2],'m--', label = 'Machine 3 SN PINN')

#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('cycles')
plt.ylabel('damage')
plt.grid(which = 'both')
plt.legend(loc=0, facecolor = 'w')

fig  = plt.figure(2)
fig.clf()

plt.plot(1/dhistall[:,-1], P,'b-', label = 'SN model')
plt.plot(1/results[:,-1], P,'r--', label = 'PINN')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('cycles')
plt.ylabel('load')
plt.grid(which = 'both')
plt.legend(loc=0, facecolor = 'w')
