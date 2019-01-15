import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as matplotlib

import sys
sys.path.append('../../../')

from model import create_model

if __name__ == "__main__":
    
    """---------------------------------------------------------------------"""
    myDtype = tf.float32
    
    "-------------------------------------------------------------------------"
    # preliminaries
    a0   = 0.005       # initial crack length [m]
    m    = 3.8         # Paris model exponent
    C    = 1.5E-11     # Paris model constant
    F    = 1.0
    aMax = 0.05

    "-------------------------------------------------------------------------"
    # fleet information
    df = pd.read_csv('aFleet_5yrs.csv', index_col = None)
    aFleet = np.asarray(df)
    
    df = pd.read_csv('SFleet_5yrs.csv', index_col = None)
    SFleet = np.transpose(np.asarray(df))

    nFleet, nCycles = SFleet.shape
    
    "-------------------------------------------------------------------------"
    SFleet = SFleet[:,:,np.newaxis]
    SFleet = ops.convert_to_tensor(SFleet, dtype = myDtype)
    
    batch_input_shape = SFleet.shape

    a0RNN = ops.convert_to_tensor(a0 * np.ones((nFleet, 1)), dtype=myDtype)
    
    "-------------------------------------------------------------------------"
    dkLayer = tf.keras.models.load_model('DK_MLP.h5')
    dkLayer.trainable = True

    modelBefore = create_model(dkLayer = dkLayer, C = C, m = m,
                         batch_input_shape = batch_input_shape, a0RNN = a0RNN,
                         myDtype = myDtype, return_sequences = True)

    aRNNbefore = modelBefore.predict_on_batch(SFleet)[:,:,0].transpose()
    errBefore  = aFleet[-1,:] - aRNNbefore[-1,:]
    mseBefore  = np.mean(errBefore**2.0)
    maeBefore  = np.max(np.abs(errBefore))
    
    model_dir = "./training_60_points"
    weight_path = model_dir + "/cp.ckpt"
    
    modelAfter = create_model(dkLayer = dkLayer, C = C, m = m,
                         batch_input_shape = batch_input_shape, a0RNN = a0RNN,
                         myDtype = myDtype, return_sequences = True)
    modelAfter.load_weights(weight_path)

    aRNNafter = modelAfter.predict_on_batch(SFleet)[:,:,0].transpose()
    errAfter  = aFleet[-1,:] - aRNNafter[-1,:]
    mseAfter  = np.mean(errAfter**2.0)
    maeAfter  = np.max(np.abs(errAfter))
    
    "-------------------------------------------------------------------------"
    yLB = 0.0
    yUB = 0.06

    "-------------------------------------------------------------------------"
    matplotlib.rc('font', size=14)
    ifig = 0

    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()
    
    strBefore = "before training\nMSE = %1.1e\nMAE = %1.1e" % (mseBefore, maeBefore)
    strAfter  = "after training\nMSE = %1.1e\nMAE = %1.1e" % (mseAfter, maeAfter)
    
    plt.plot([yLB, yUB], [yLB, yUB], '--k')
    plt.plot(aFleet[-1,:], aRNNbefore[-1,:], 'o', label = strBefore)
    plt.plot(aFleet[-1,:], aRNNafter[-1,:], 'o', label = strAfter)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.xlim(yLB, yUB)
    plt.ylim(yLB, yUB)
    plt.grid(which='both')
    plt.legend(loc = 'lower right',facecolor = 'w')
    