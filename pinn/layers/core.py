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

class pmlDenseLayer(Layer):
    def __init__(self, num_outputs, kernel_initializer = 'glorot_uniform', **kwargs):
        super(pmlDenseLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = initializers.get(kernel_initializer)
    
    def build(self, input_shape, **kwargs):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.kernel = self.add_weight("kernel",
                                      shape = [input_shape[-1].value, self.num_outputs],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = True,
                                      **kwargs)
        self.built = True
    
    def call(self, inputs):
        inputs  = ops.convert_to_tensor(inputs, dtype=self.dtype)
        return gen_math_ops.mat_mul(inputs, self.kernel)
    
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.num_outputs)
    
    def get_config(self):
        config = {
                'num_outputs': self.num_outputs,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                }
        base_config = super(pmlDenseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
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
