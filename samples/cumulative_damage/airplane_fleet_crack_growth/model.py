import numpy as np

import tensorflow as tf
import sys
sys.path.append('../../../')

from pinn.layers import ParisLaw, CumulativeDamageCell

def create_model(dkLayer, C, m, batch_input_shape, a0RNN, myDtype, return_sequences = False, unroll = False):
    
    da_input_shape = dkLayer.get_output_shape_at(-1)
    daLayer = ParisLaw(input_shape = da_input_shape, dtype = myDtype)
    daLayer.build(input_shape = da_input_shape)
    daLayer.set_weights([np.asarray([C, m], dtype=daLayer.dtype)])
    daLayer.trainable = False
    	
    PINNhybrid = tf.keras.Sequential()
    PINNhybrid.add(dkLayer)
    PINNhybrid.add(daLayer)

    "-------------------------------------------------------------------------"
    CDMCellHybrid = CumulativeDamageCell(model = PINNhybrid,
                                       batch_input_shape = batch_input_shape,
                                       dtype = myDtype,
                                       initial_damage = a0RNN)
     
    CDMRNNhybrid = tf.keras.layers.RNN(cell = CDMCellHybrid,
                                       return_sequences = return_sequences,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll = unroll)

    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])
    
    return model
