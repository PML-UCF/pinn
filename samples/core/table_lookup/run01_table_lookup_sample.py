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
import pandas as pd

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

import sys
sys.path.append('../../../')
from tensorflow.contrib.image.python.ops.dense_image_warp import _interpolate_bilinear as interpolate

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

def create_model(grid_array, bounds, input_shape, table_shape):
    dLInterpol = tableInterpolation(input_shape = input_shape)
    dLInterpol.build(input_shape = table_shape)
    dLInterpol.set_weights([grid_array, bounds])
    model = tf.keras.Sequential()
    model.add(dLInterpol)
    return model

myDtype = tf.float32
#df = pd.read_csv('aSKF_kappa1.csv')
#data = np.transpose(np.repeat(np.asarray([df['askf']]),len(df['askf']),axis=0))
#space = np.asarray([df['xval'],np.ones(len(df['xval']))])
df = pd.read_csv('aSKF_kappa12.csv')
data = np.transpose(np.asarray(np.transpose(df))[1:])
if data.shape[1] == 1:
    data = np.repeat(data,2,axis=1)
space = np.asarray([np.asarray(df['xval']),np.asarray([float(i) for i in df.columns[1:]])])
table_shape = (data.shape[0],2)

if space.shape[0] == 1:
    bounds = np.asarray([[np.min(space[0]),np.max(space[0])]])
elif space.shape[0] == 2:
    bounds = np.asarray([[np.min(space[0]),np.max(space[0])],[np.min(space[1]),np.max(space[1])]])
    
q = np.asarray([[0.05,1.5],[0.1,1.0],[0.3,2.5],[2.0,2.0]])
input_array = ops.convert_to_tensor(q,dtype=tf.float32)
input_shape = input_array.shape
input_array= tf.expand_dims(input_array,0)

model = create_model(data, bounds, input_shape, table_shape)
result = model.predict(input_array, steps = 1)
print(result)
