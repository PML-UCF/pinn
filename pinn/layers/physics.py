import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import Dense

from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import common_shapes

from tensorflow.linalg import diag as tfDiag
from tensorflow.math import reciprocal

class StressIntensityRange(Layer):
    """Just your regular stress intensity range implementation.
    `StressIntensityRange` implements the operation:
        `output = F*input[0]*sqrt(pi*input[1])
        where:
            * `F` is a dimensionless function of geometry and the relative crack length,        
            * input[:,0] is the crack length, and
            * input[:,1] is the nominal stress.
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
#        input_dim = input_shape[-1]
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
#        self.input_dim   = input_dim
        self.built = True

    def call(self, inputs):
        inputs  = ops.convert_to_tensor(inputs, dtype=self.dtype)
        if common_shapes.rank(inputs) is not 2:
            raise ValueError('`StressIntensityRange` only takes "rank 2" inputs.')
        output = self.kernel*inputs[:,1]*gen_math_ops.sqrt(np.pi*inputs[:,0])
        return output
#        return array_ops.reshape(output, (output.shape[0],1))
    
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
#        return array_ops.reshape(output, (output.shape[0],1))
    
    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 
		