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

""" Input selection sample case
"""
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import ops

import sys
sys.path.append('../')
from pinn.layers import CumulativeDamageCell
# =============================================================================
# Function
# =============================================================================
class CrackThreshold(Layer):
    """ 
        `output = sig*inputs[:,2]+(1-sig)*inputs[:,1]`
        where:
            * `sig` is the response of the sigmoid function used to filter between 
                    initiation and propagation mechanisms,
            * inputs[:,0] is current crack length of the previous time step,
            * inputs[:,1] is crack length variation for the initiation stage,
            * inputs[:,2] is crack length variation for the propagation stage,
            * output is the overall crack length variation         
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(CrackThreshold, self).__init__(**kwargs)
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
        sig = 1/(1+gen_math_ops.exp(-self.kernel[0]*(inputs[:,0]-self.kernel[1])))
        output = sig*inputs[:,2]+(1-sig)*inputs[:,1]
        if(output.shape[0].value is not None):
            output = tf.reshape(output, (tensor_shape.TensorShape((output.shape[0],1))))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 
    
def create_model(alpha, ath, batch_input_shape, da0RNN, myDtype, return_sequences = False, unroll = False):
    thLayer = CrackThreshold(input_shape = batch_input_shape, dtype = myDtype)
    thLayer.build(input_shape = batch_input_shape)
    thLayer.set_weights([np.asarray([alpha,ath], dtype = thLayer.dtype)])
    thLayer.trainable = False
    
# =============================================================================
#     model = tf.keras.Sequential()
#     model.add(thLayer)
# =============================================================================
    
    PINN = tf.keras.Sequential()
    PINN.add(thLayer)
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

#--------------------------------------------------------------------------
if __name__ == "__main__":
    myDtype = tf.float32
    
    ath = .5e-3
    alpha = 1e6
        
# =============================================================================
#     a = np.linspace(0,1e-3,100)
#     dai = .25e-3*np.ones(len(a))
#     dap = .75e-3*np.ones(len(a))
#     
#     input_array = np.asarray([a,dai,dap])
#     input_array = np.transpose(input_array)
#     input_array = input_array.reshape((1,len(a),3))
#     batch_input_shape = input_array.shape
#     
#     model = create_model(alpha = alpha, ath = ath, batch_input_shape = batch_input_shape, 
#                          myDtype = myDtype)
#     results = model.predict_on_batch(input_array)
# =============================================================================
    np.random.seed(123)
    
    aux = np.random.random((10,15))
    
    dai = .025*ath*aux
    dap = .075*ath*aux
    
    inputs = np.asarray([dai,dap])
    inputs = inputs.reshape(np.shape(inputs)[1],np.shape(inputs)[2],2)
    
    batch_input_shape = inputs.shape
    ao = 1e-9
    #--------------------------------------------------------------------------
    # Prediction sequence
    da0RNN = ops.convert_to_tensor(ao * np.ones((inputs.shape[0], 1)), dtype=myDtype)
        
    model = create_model(alpha = alpha, ath = ath, batch_input_shape = batch_input_shape, 
                         da0RNN = da0RNN, myDtype = myDtype, return_sequences = True, unroll = True)
    
    results = model.predict_on_batch(inputs)
    #--------------------------------------------------------------------------    
    # Plot delta crack history for a single machines
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(results[:,-1]*1e3,'-k')
    plt.plot(dai[:,-1]*1e3,':r', label = 'init.')
    plt.plot(dap[:,-1]*1e3,':b', label = 'prop.')
    
    plt.title('Threshold response (tf Layer)')
    plt.xlabel('machine')
    plt.ylabel('$\Delta$ a [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')
    

