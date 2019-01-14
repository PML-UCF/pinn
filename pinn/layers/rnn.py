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

class CumulativeDamageCell(Layer):
    """Cummulative damage implementation.
    """

    """
    #TypeError: The argument 'cell' (<src.pmlPILayers.CumulativeDamageCell object at 0x0000013D4FD9A9B0>)
    is not an RNNCell: 'output_size' property is missing, 'state_size' property is missing,
    either 'zero_state' or 'get_initial_state' method is required.
    """
    def __init__(self, model, units = 1, initial_damage = None, **kwargs):
        super(CumulativeDamageCell, self).__init__(**kwargs)
        self.units = units
        self.model = model
        self.initial_damage = initial_damage
        self.state_size  = tensor_shape.TensorShape(self.units)
        self.output_size = tensor_shape.TensorShape(self.units)

    def build(self, input_shape, **kwargs):
        for var in self.model.weights:
            if var.trainable:
                self._trainable_weights.append(var)
            else:
                self._non_trainable_weights.append(var)

        self.built = True

    def call(self, inputs, states):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        states = ops.convert_to_tensor(states, dtype=self.dtype)
        
        states = states[0,:]
        x_d_tm1 = array_ops.concat((states, inputs), axis = 1)
        da_t = self.model(x_d_tm1)
        d = da_t + states
        
        return d, [d]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self.initial_damage is None:
            initial_state = _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)
        else:
            initial_state = self.initial_damage
            
        return initial_state
    
    
"""----------------------------------------------------------------------------
#friend functions
#----------------------------------------------------------------------------"""
def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

#def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
#    if inputs is not None:
#        batch_size = array_ops.shape(inputs)[0]
#        dtype = inputs.dtype
#    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)
#
def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
   """Generate a zero filled tensor with shape [batch_size, state_size]."""
   if None in [batch_size_tensor, dtype]:
       raise ValueError(
               'batch_size and dtype cannot be None while constructing initial state: '
               'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))
   if _is_multiple_state(state_size):
       states = []
       for dims in state_size:
           flat_dims = tensor_shape.as_shape(dims).as_list()
           init_state_size = [batch_size_tensor] + flat_dims
           init_state = array_ops.zeros(init_state_size, dtype=dtype)
           states.append(init_state)
       return states
   else:
       flat_dims = tensor_shape.as_shape(state_size).as_list()
       init_state_size = [batch_size_tensor] + flat_dims
       return array_ops.zeros(init_state_size, dtype=dtype)

def _is_multiple_state(state_size):
   """Check whether the state_size contains multiple states."""
   return (hasattr(state_size, '__len__') and
           not isinstance(state_size, tensor_shape.TensorShape))
