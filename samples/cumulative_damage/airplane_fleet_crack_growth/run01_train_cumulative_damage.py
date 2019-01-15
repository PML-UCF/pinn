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

""" Train physics-informed recursive neural network
"""

import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np
import pandas as pd

import sys
sys.path.append('../../../')

from model import create_model


if __name__ == "__main__":

    #--------------------------------------------------------------------------
    # preliminaries
    myDtype = tf.float32

    a0   = 0.005       # initial crack length [m]
    m    = 3.8         # Paris model exponent
    C    = 1.5E-11     # Paris model constant

    #--------------------------------------------------------------------------
    # fleet information
    df = pd.read_csv('aFleet_5yrs.csv', index_col = None)
    aFleet = np.asarray(df)
    
    df = pd.read_csv('SFleet_5yrs.csv', index_col = None)
    SFleet = np.transpose(np.asarray(df))
    nFleet, nCycles = SFleet.shape
    
    #--------------------------------------------------------------------------
    idx = np.argsort(aFleet[-1,:])

    arange = np.asarray(np.linspace(0,299, 60), dtype = int)
    idxTrain = idx[arange]
    
    Sobs = SFleet[idxTrain,:]
    Sobs = Sobs[:,:,np.newaxis]
    Sobs = ops.convert_to_tensor(Sobs, dtype = myDtype)
    
    batch_input_shape = Sobs.shape

    SFleet = SFleet[:,:,np.newaxis]
    SFleet = ops.convert_to_tensor(SFleet, dtype = myDtype)

    nObs = Sobs.shape[0]

    #--------------------------------------------------------------------------
    aTarget = aFleet[-1, idxTrain]
    aTarget = aTarget[:, np.newaxis]
    aTarget = ops.convert_to_tensor(aTarget, dtype=myDtype)

    a0RNN = ops.convert_to_tensor(a0 * np.ones((nObs, 1)), dtype=myDtype)
    
    #--------------------------------------------------------------------------
    dkLayer = tf.keras.models.load_model('DK_MLP.h5')
    dkLayer.trainable = True

    model = create_model(dkLayer = dkLayer, C = C, m = m,
                         batch_input_shape = batch_input_shape, a0RNN = a0RNN, myDtype = myDtype)
    
    #--------------------------------------------------------------------------
    EPOCHS = 20
    jmdDir = "./training_%d_points" % len(idxTrain)
    weight_path = jmdDir + "/cp.ckpt"
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath = weight_path, monitor = 'loss',
                                                    verbose = 1, save_best_only = True,
                                                    mode = 'min', save_weights_only = False)
    
    CSVLogger = tf.keras.callbacks.CSVLogger(filename = jmdDir + "/training.log", append = False)
    
    ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
                                       min_lr = 1e-15, patience=10, verbose=1, mode='min')
    
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", verbose=2,
                          patience=10)
    
    TensorBoard = tf.keras.callbacks.TensorBoard(log_dir= jmdDir +  "/logs")

    callbacks_list = [ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard]

    history = model.fit(Sobs, aTarget, epochs=EPOCHS, steps_per_epoch=1, verbose=1, callbacks=callbacks_list)
    
    #--------------------------------------------------------------------------
    df = pd.DataFrame.from_dict(history.history)
    df.insert(loc = 0, column='epoch', value = history.epoch)
    df.to_csv(jmdDir + "/lossHistory.csv", index = False)
    