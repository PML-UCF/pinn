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

""" Physics-informed layers
"""

import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer

#TODO: addept to tf2
from tensorflow.compat.v1 import placeholder

from tensorflow.python.ops import gen_math_ops, array_ops

from tensorflow import reshape

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import common_shapes


class StressIntensityRange(Layer):
    """Just your regular stress intensity range implementation.
    `StressIntensityRange` implements the operation:
        `output = F*input[:,1]*sqrt(pi*input[:,0])
        where:
            * `F` is a dimensionless function of geometry and the relative crack length,        
            * input[:,0] is the crack length, and
            * input[:,1] is the nominal stress range.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            
        super(StressIntensityRange, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)

        self.kernel = self.add_weight("kernel",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype = self.dtype,
                                      trainable = True)
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        if common_shapes.rank(inputs) is not 2: 
            raise ValueError('`StressIntensityRange` only takes "rank 2" inputs.')

        output = gen_math_ops.mul(self.kernel*inputs[:,1], gen_math_ops.sqrt(np.pi*inputs[:, 0]))
        output = array_ops.reshape(output, (array_ops.shape(output)[0], 1))

        # outputs should be (None, 1), so it is still rank = 2
        return output
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1)


class ParisLaw(Layer):
    """Just your regular Paris law implementation.
    `ParisLaw` implements the operation:
        `output = C*(input**m)`
        where `C` and `m` are the Paris law constants.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ParisLaw, self).__init__(**kwargs)
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
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        if rank is not 2:
            raise ValueError('`ParisLaw` only takes "rank 2" inputs.')
        output = self.kernel[0]*(inputs**self.kernel[1])
        return output
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 


class SNCurve(Layer):
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
        super(SNCurve, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape = [2],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = 1/10**(self.kernel[0]*inputs+self.kernel[1])
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 


class WalkerModel(Layer):
    """A modified version of Paris law to take into account the stress ratio effect.
    `WalkerModel` implements the operation:
        `output = C*(inputs[:,0]**m)`
        where `C` and `m` are constants, and `C` is obtained from the following
        relation:
            `C = Co/((1-inputs[:,1])**(m*(1-gamma))))`
            
            * input[:,0] is the nominal stress range
            * input[:,1] is the stress ratio, and
            
            * sig is a custumized sigmoid function to calibrate Walker's coefficient (gamma)
                  with respect to the stress ratio value.        
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(WalkerModel, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape = [4],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.built = True
        
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        if common_shapes.rank(inputs) is not 2: 
            raise ValueError('`WalkerModel` only takes "rank 2" inputs.')
 
        sig = 1/(1+gen_math_ops.exp(self.kernel[0]*inputs[:,1]))
        gamma = sig*self.kernel[1]
        C = self.kernel[2]/((1-inputs[:,1])**(self.kernel[3]*(1-gamma)))
        output = C*(inputs[:,0]**self.kernel[3])
        output = array_ops.reshape(output,(array_ops.shape(output)[0],1))
        return output
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1)
    
