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

""" RNN layers
"""

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape


class CumulativeDamageCell(Layer):
    """Cummulative damage cell implementation.
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
            initial_state = ops.convert_to_tensor(self.initial_damage, dtype=self.dtype)
            
        return initial_state
    
#------------------------------------------------------------------------------    
#friend functions
#------------------------------------------------------------------------------    
def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

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
