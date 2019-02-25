# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:33:53 2019

@author: ar679403
"""

""" Physics-informed layers
"""

import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import tensor_shape
# =============================================================================
# Classes
# =============================================================================
class EquivalentStress(Layer):
    """Equivalent stress required by SN curves:
        `output = inputs[:,1]*((1-inputs[:,2]/inputs[:,1])**c)
        where:
            * `c` is a dimensionless constant,        
            * inputs[:,1] maximum stress, and
            * inputs[:,2] minimum stress.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            
        super(EquivalentStress, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `EquivalentStress` '
                             'should be defined. Found `None`.')
        self.kernel = self.add_weight("kernel",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype = self.dtype,
                                      trainable = True)
        self.built = True

    def call(self, inputs):
        inputs  = ops.convert_to_tensor(inputs, dtype=self.dtype)
        output = inputs[:,1]*((1-inputs[:,2]/inputs[:,1])**self.kernel)
        return output
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1)

class InitiationSN(Layer):
    """Alloy SN curve:
        `output = c[3]*(1/10**(c[0]+c[1]*np.log10(inputs-c[2])))
        where:
            * `c` Dimensionless constants,        
            * inputs Equivalent stress.
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(InitiationSN, self).__init__(**kwargs)
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
        output = self.kernel[3]*(1/10**(self.kernel[0]+self.kernel[1]*np.log10(inputs[:,1]-self.kernel[2]))) 
        return output
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 

class StressIntensityRange(Layer):
    """Just your regular stress intensity range implementation.
    `StressIntensityRange` implements the operation:
        `output = F*(input[:,0]-input[:,1])*sqrt(pi*input[:,2])
        where:
            * `F` is a dimensionless function of geometry and the relative crack length,        
            * input[:,1] is the maximum stress
            * input[:,2] is the minimum stress
            * input[:,0] is the crack length.
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
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `StressIntensityRange` '
                             'should be defined. Found `None`.')
        self.kernel = self.add_weight("kernel",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype = self.dtype,
                                      trainable = True)
        self.built = True

    def call(self, inputs):
        inputs  = ops.convert_to_tensor(inputs, dtype=self.dtype)
        output = self.kernel*(inputs[:,1]-inputs[:,2])*gen_math_ops.sqrt(np.pi*inputs[:,0])
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
        output = self.kernel[0]*(inputs**self.kernel[1])
        return output
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 