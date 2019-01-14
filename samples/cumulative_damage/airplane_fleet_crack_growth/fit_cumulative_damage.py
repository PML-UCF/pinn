#import numpy as np
#import pandas as pd
#
#import tensorflow as tf
#from tensorflow.python.framework import ops

import sys

sys.path.append('../../../')

from pinn.layers import StressIntensityRange, ParisLaw, CumulativeDamageCell

#import pinn.layers as pl

if __name__ == "__main__":
    a = 1.0
#    """---------------------------------------------------------------------"""
#    myDtype = tf.float32
#    
#    "-------------------------------------------------------------------------"
#    # preliminaries
#    a0   = 0.005       # initial crack length [m]
#    m    = 3.8         # Paris model exponent
#    C    = 1.5E-11     # Paris model constant
#    F    = 1.0
#    aMax = 0.05
#
#    "-------------------------------------------------------------------------"
#    # fleet information
#    df = pd.read_csv('aFleet_5yrs.csv', index_col = None)
#    aFleet = np.asarray(df)
#    
#    df = pd.read_csv('SFleet_5yrs.csv', index_col = None)
#    SFleet = np.transpose(np.asarray(df))
#
#    nFleet, nCycles = SFleet.shape
#    
#    "-------------------------------------------------------------------------"
#    idx = np.argsort(aFleet[-1,:])
##    arange = np.asarray(np.linspace(0,299, 5), dtype = int)
##    arange = np.asarray(np.linspace(0,299, 15), dtype = int)
##    arange = np.asarray(np.linspace(0,299, 30), dtype = int)
#    arange = np.asarray(np.linspace(0,299, 60), dtype = int)
#    
#    idxTrain = idx[arange]
##    idxTrain = idx[np.arange(0,nFleet,20)]
##    idxTrain = idx[np.arange(0,nFleet,10)]
##    idxTrain = idx[np.arange(0,nFleet,5)]
#    
##    idxFleet = np.arange(0,nFleet)
##    idxTrain, idxTest, _, _ = train_test_split(idxFleet, idxFleet, test_size = 0.8)
#    
#    Sobs = SFleet[idxTrain,:]
#    Sobs = Sobs[:,:,np.newaxis]
#    Sobs = ops.convert_to_tensor(Sobs, dtype = myDtype)
#
#    SFleet = SFleet[:,:,np.newaxis]
#    SFleet = ops.convert_to_tensor(SFleet, dtype = myDtype)
#
#    nAssets = Sobs.shape[0]
#
#    "-------------------------------------------------------------------------"
#    aTarget = aFleet[-1, idxTrain]
#    aTarget = aTarget[:, np.newaxis]
#    aTarget = ops.convert_to_tensor(aTarget, dtype=myDtype)
#
#    a0RNN = ops.convert_to_tensor(a0 * np.ones((nAssets, 1)), dtype=myDtype)
#    
#    "-------------------------------------------------------------------------"
#    dkLayer = tf.keras.models.load_model('MLP_DK_02.h5')
#    dkLayer.trainable = True
#
#    da_input_shape = dkLayer.get_output_shape_at(-1)
#    daLayer = ParisLaw(input_shape = da_input_shape, dtype = myDtype)
#    daLayer.build(input_shape = da_input_shape)
#    daLayer.set_weights([np.asarray([C, m], dtype=daLayer.dtype)])
#    daLayer.trainable = False
#    	
#    PINNhybrid = tf.keras.Sequential()
#    PINNhybrid.add(dkLayer)
#    PINNhybrid.add(daLayer)
#
#    "-------------------------------------------------------------------------"
#    CDMCellHybrid = CumulativeDamageCell(model = PINNhybrid,
#                                       batch_input_shape = Sobs.shape,
#                                       dtype = myDtype,
#                                       initial_damage = a0RNN)
#     
#    CDMRNNhybrid = tf.keras.layers.RNN(cell = CDMCellHybrid,
#                                       return_sequences = False,
#                                       return_state = False,
#                                       batch_input_shape = Sobs.shape,
#                                       unroll=False)
#
#    model = tf.keras.Sequential()
#    model.add(CDMRNNhybrid)
#    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])
#    
#    "-------------------------------------------------------------------------"
#    EPOCHS = 20 #int(1e4)
#    jmdDir = "./jmd_training_loss_mse_%d_points" % len(idxTrain)
#    weight_path = jmdDir + "/cp.ckpt"
#    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath = weight_path, monitor = 'loss',
#                                                    verbose = 1, save_best_only = True,
#                                                    mode = 'min', save_weights_only = False)
#    
#    CSVLogger = tf.keras.callbacks.CSVLogger(filename = jmdDir + "/training.log", append = False)
#    
#    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
#                                       min_lr = 1e-15, patience=10, verbose=1, mode='min')
#    
#    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", verbose=2,
#                          patience=10)
#    
#    TensorBoard = tf.keras.callbacks.TensorBoard(log_dir= jmdDir +  "/logs")
#
#    callbacks_list = [ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard]
#
#    history = model.fit(Sobs, aTarget, epochs=EPOCHS, steps_per_epoch=1, verbose=1, callbacks=callbacks_list)
#    
#    
#    df = pd.DataFrame.from_dict(history.history)
#    df.insert(loc = 0, column='epoch', value = history.epoch)
#    df.to_csv(jmdDir + "/lossHistory.csv", index = False)
#    