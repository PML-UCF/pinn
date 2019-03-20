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

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import tensor_shape

from tensorflow.contrib.image.python.ops.dense_image_warp import _interpolate_bilinear as interpolate

import numpy as np
import tensorflow as tf

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

def inputsSelection(inputs, ndex):
    input_mask = np.zeros([inputs.shape[-1], len(ndex)])
    for i in range(inputs.shape[-1]):
        for v in ndex:
            if i == v:
                input_mask[i,np.where(ndex == v)] = 1
        
    dL = Dense(len(ndex), activation = None, input_shape = inputs.shape, 
               use_bias = False)
    dL.build(input_shape = inputs.shape)
    dL.set_weights([input_mask])
    dL.trainable = False
    return dL
 
class tableInterpolation(Layer):
    """ Table lookup and interpolation implementation.
        Interrogates provided query points using provided table and outputs the interpolation result.
        Remarks on this class:
            - Only supports 2-D tables (f(x1,x2) = y)
            - If a 1-D table is to be used, it needs to be converted to a 2-D table (see file /samples/core/table_lookup/run01_table_lookup_sample.py)
            - Extrapolation is not supported (provide a table grid large enough for your case)
            - Class returns limit values in case of extrapolation attempt.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(tableInterpolation, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.grid = self.add_weight("grid",
                                      shape = input_shape,
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.bounds = self.add_weight("bounds",
                                      shape = [2,2],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        self.grid = tf.expand_dims(self.grid,0)
        self.grid = tf.expand_dims(self.grid,-1)
        self.bounds = ops.convert_to_tensor(self.bounds,dtype=tf.float32)
        queryPointsX_ind = (tf.to_float(tf.shape(self.grid)[1])-tf.constant(1.0))*(tf.transpose(inputs[0])[0]-self.bounds[0][0])/(self.bounds[0][1]-self.bounds[0][0])
        queryPointsV_ind = (tf.to_float(tf.shape(self.grid)[2])-tf.constant(1.0))*(tf.transpose(inputs[0])[1]-self.bounds[1][0])/(self.bounds[1][1]-self.bounds[1][0])
        queryPoints_ind = tf.stack([queryPointsX_ind,queryPointsV_ind],1)
        queryPoints_ind = tf.expand_dims(queryPoints_ind,0)
        output = interpolate(self.grid, queryPoints_ind)
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 