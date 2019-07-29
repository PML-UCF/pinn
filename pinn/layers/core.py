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
from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops

from pinn.layers import interpolate

from tensorflow import shape, expand_dims, constant, cast

import numpy as np


def getScalingDenseLayer(input_location, input_scale):
    recip_input_scale = np.reciprocal(input_scale)
    
    waux = np.diag(recip_input_scale)
    baux = -input_location*recip_input_scale
    
    dL = Dense(input_location.shape[0], activation = None, input_shape = input_location.shape)
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


class TableInterpolation(Layer):
    """ Table lookup and interpolation implementation.
        Interrogates provided query points using provided table and outputs the interpolation result.
        Remarks on this class:
            - Only supports 2-D tables (f(x1,x2) = y)
            - If a 1-D table is to be used, it needs to be converted to a 2-D table (see file /samples/core/table_lookup/run01_table_lookup_sample.py)
            - Extrapolation is not supported (provide a table grid large enough for your case)
            - Class returns limit values in case of extrapolation attempt.
            - Provided tables should be equally spaced.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 table_shape=(1,4,4,1),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TableInterpolation, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
        self.table_shape = table_shape
        
    def build(self, input_shape, **kwargs):
        self.grid = self.add_weight("grid",
                                      shape = self.table_shape,
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        self.bounds = self.add_weight("bounds",
                                      shape = [2,2],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        self.grid = ops.convert_to_tensor(self.grid, dtype=self.dtype)
        self.bounds = ops.convert_to_tensor(self.bounds,dtype=self.dtype)
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        queryPoints_ind = ((cast(shape(self.grid)[1:3], dtype=self.dtype))-constant(1.0))*(inputs-self.bounds[0])/(self.bounds[1]-self.bounds[0])
        if common_shapes.rank(inputs) == 2:
            queryPoints_ind = expand_dims(queryPoints_ind,0)
        output = interpolate(self.grid, queryPoints_ind)
        if common_shapes.rank(inputs) == 2:
            output = array_ops.reshape(output,(array_ops.shape(output)[1],) + (array_ops.shape(output)[2],))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        
        return aux_shape[:-1].concatenate(1)
