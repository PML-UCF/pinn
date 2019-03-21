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
""" Core PINN layers
"""
from tensorflow.keras.layers import Dense
from tensorflow.python.framework import ops

from tensorflow.linalg import diag as tfDiag
from tensorflow.math import reciprocal

import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops

from tensorflow import reshape, placeholder

def getScalingDenseLayer(input_location, input_scale, dtype):
    input_location    = ops.convert_to_tensor(input_location, dtype=dtype)
    input_scale       = ops.convert_to_tensor(input_scale, dtype=dtype)
    recip_input_scale = reciprocal(input_scale)
    
    waux = tfDiag(recip_input_scale)
    baux = -input_location*recip_input_scale
    
    dL = Dense(input_location.get_shape()[0], activation = None, input_shape = input_location.shape)
    dL.build(input_shape = input_location.shape)
    dL.set_weights([waux, baux])
    dL.trainable = False
    return dL

def inputsSelection(inputs_shape, ndex):
    if not hasattr(ndex,'index'):
        ndex = list(ndex)
    input_mask = np.zeros([inputs_shape[-1], len(ndex)])
    for i in range(inputs_shape[-1]):
        for v in ndex:
            if i == v:
                input_mask[i,ndex.index(v)] = 1
        
    dL = Dense(len(ndex), activation = None, input_shape = inputs_shape, 
               use_bias = False)
    dL.build(input_shape = inputs_shape)
    dL.set_weights([input_mask])
    dL.trainable = False
    return dL

class SigmoidSelector(Layer):
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
        super(SigmoidSelector, self).__init__(**kwargs)
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
        if inputs.shape[0].value is not None:
            sig = 1/(1+gen_math_ops.exp(-self.kernel[0]*(inputs[:,0]-self.kernel[1])))
            output = sig*inputs[:,2]+(1-sig)*inputs[:,1]
            output = reshape(output, (tensor_shape.TensorShape((output.shape[0],1))))
        else:
            output = placeholder(dtype=self.dtype,
                                 shape=tensor_shape.TensorShape([inputs.shape[0],1]))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1)
      