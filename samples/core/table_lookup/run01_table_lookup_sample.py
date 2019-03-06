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

""" Table lookup interpolation sample case
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow import reshape, placeholder

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.ops import gen_math_ops

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import common_shapes

from tensorflow.python.keras.engine.base_layer import InputSpec
import sys
sys.path.append('../../../')
from tensorflow.keras.layers import Dense
from tensorflow.contrib.image.python.ops.dense_image_warp import _interpolate_bilinear as interp

class tableInterpolation(Layer):
    """ SN-Curve implementation (REF: https://en.wikipedia.org/wiki/Fatigue_(material)#Stress-cycle_(S-N)_curve)
        `output = 1/10**(a*inputs+b)`
        where:
            * `a`,`b` parametric constants for linear curve,
            * input is cyclic stress, load, or temperature (depends on the application) in log10 space,
            * output is delta damage
        Notes:
            * This layer represents SN-Curve linearized in log10-log10 space
            * (a*inputs+b) expression gives number of cycles in log10 space corresponding to stress level
        Linearization:
            * For an SN-Curve with an equation of N = C1*(S**C2) , take log10 of both sides
            * log10(N) = log10(C1) + C2*log10(S), yields to:
                C2 = a
                log10(C1) = b
                log10(S) = inputs            
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
        self.grid = self.add_weight("kernel",
                                      shape = [3,3],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.bounds = self.add_weight("bias",
                                      shape = [2,2],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        self.grid = tf.expand_dims(self.grid,0)
        self.grid = tf.expand_dims(self.grid,-1)
        queryPointsX_ind = (self.grid.shape[0]-1)*(inputs[0]-self.bounds[0][0])/(self.bounds[0][1]-self.bounds[0][0])
        queryPointsV_ind = (self.grid.shape[1]-1)*(inputs[1]-self.bounds[1][0])/(self.bounds[1][1]-self.bounds[1][0])
        queryPoints_ind = tf.stack([queryPointsX_ind,queryPointsV_ind],1)
        queryPoints_ind = tf.expand_dims(queryPoints_ind,0)
        output = interp(self.grid, queryPoints_ind)
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 

def create_model(grid_array, bounds, input_shape):
    dLInterpol = tableInterpolation(input_shape = input_shape)
    dLInterpol.build(input_shape = input_shape)
    dLInterpol.set_weights([grid_array, bounds])
    model = tf.keras.Sequential()
    model.add(dLInterpol)
    return model

myDtype = tf.float32
data = np.asarray([[0,0,0],[1,2,3],[10,20,30]])
space = np.asarray([[500,750,1000],[4,6,8]])
bounds = np.asarray([[np.min(space[0]),np.max(space[0])],[np.min(space[1]),np.max(space[1])]])
#data = data[np.newaxis,:,:,np.newaxis]
grid = ops.convert_to_tensor(data,dtype=tf.float32)
q = np.asarray([[0.0,0.0],[1.5,1.0],[2.0,2.0]])

#q = q[np.newaxis,:,:]
input_array = ops.convert_to_tensor(q,dtype=tf.float32)
input_shape = input_array.shape

model = create_model(grid, bounds, input_shape)
result = model.predict(input_array)
print(result)
