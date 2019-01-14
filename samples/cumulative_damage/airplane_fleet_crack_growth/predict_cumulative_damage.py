import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as matplotlib

import tensorflow as tf
from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

from pmlPILayers import StressIntensityRange, ParisLaw, CumulativeDamageCell

def create_model(dkLayerModel, C, m, batch_input_shape, a0RNN, myDtype):
    dkLayer = tf.keras.models.load_model(dkLayerModel)
    dkLayer.trainable = True

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
                                       return_sequences = True,
                                       return_state = False,
                                       batch_input_shape = batch_input_shape,
                                       unroll=False)

    model = tf.keras.Sequential()
    model.add(CDMRNNhybrid)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(1e-12), metrics=['mae'])
    
    return model


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
    flagCase = 0
    
    if flagCase == 0:
        df = pd.read_csv('aFleet_5yrs.csv', index_col = None)
        aFleet = np.asarray(df)
        df = pd.read_csv('SFleet_5yrs.csv', index_col = None)
        cols = df.columns
        SFleet = np.transpose(np.asarray(df))
    else:
        df = pd.read_csv('aFleet.csv', index_col = None)
        aFleet = np.asarray(df)
        df = pd.read_csv('SFleet.csv', index_col = None)
        cols = df.columns
        SFleet = np.transpose(np.asarray(df))

    nFleet, nCycles = SFleet.shape

    "-------------------------------------------------------------------------"
    idx = np.argsort(aFleet[-1,:])
#    idxTrain = idx[np.arange(0,nFleet,60)]
#    idxTrain = idx[np.arange(0,nFleet,20)]
#    idxTrain = idx[np.arange(0,nFleet,10)]
#    idxTrain = idx[np.arange(0,nFleet,5)]
    
#    arange = np.asarray(np.linspace(0,299, 5), dtype = int)
#    arange = np.asarray(np.linspace(0,299, 15), dtype = int)
#    arange = np.asarray(np.linspace(0,299, 30), dtype = int)
#    arange = np.asarray(np.linspace(0,299, 60), dtype = int)
    
#    arange = np.arange(0,15,1)
#    arange = np.arange(285,300,1)
#    idxTrain = idx[arange]
    idxTrain = idx[[0, 190, 210, 250, 260, 265, 270, 275, 280, 282, 285, 290, 292, 295, 299]]
    
    idxTest = idx[np.arange(0,nFleet,1)]
#    idxTest = np.delete(idxTest,idxTrain)
    idxPredict = idxTest.copy()

    Sobs = SFleet[idxPredict,:]
    Sobs = Sobs[:,:,np.newaxis]
    Sobs = ops.convert_to_tensor(Sobs, dtype = myDtype)

    SFleet = np.transpose(SFleet)
 #   SFleet = SFleet[:,:,np.newaxis]
 #   SFleet = ops.convert_to_tensor(SFleet, dtype = myDtype)

    nAssets = Sobs.shape[0]

    "-------------------------------------------------------------------------"
    aTarget = aFleet[-1, idxPredict]
    aTarget = aTarget[:, np.newaxis]
    aTarget = ops.convert_to_tensor(aTarget, dtype=myDtype)

    a0RNN = ops.convert_to_tensor(a0 * np.ones((nAssets, 1)), dtype=myDtype)
    
    "-------------------------------------------------------------------------"
    dkLayerModel = 'MLP_DK_02.h5'
    batch_input_shape = Sobs.shape
    modelBefore = create_model(dkLayerModel, C, m, batch_input_shape, a0RNN, myDtype)
    
    aRNNbefore = modelBefore.predict_on_batch(Sobs)[:,:,0].transpose()
    errBefore  = aFleet[-1, idxPredict] - aRNNbefore[-1,:]
    mseBefore  = np.mean(errBefore**2.0)
    maeBefore  = np.max(np.abs(errBefore))
    
    "-------------------------------------------------------------------------"
    jmdDir = "./jmd_training_loss_mse_%d_points_02" % len(idxTrain)
    weight_path = jmdDir + "/cp.ckpt"

    modelAfter = create_model(dkLayerModel, C, m, batch_input_shape, a0RNN, myDtype)
    modelAfter.load_weights(weight_path)

    aRNNafter = modelAfter.predict_on_batch(Sobs)[:,:,0].transpose()
    errAfter  = aFleet[-1, idxPredict] - aRNNafter[-1,:]
    mseAfter  = np.mean(errAfter**2.0)
    maeAfter  = np.max(np.abs(errAfter))
    
    df = pd.read_csv(jmdDir + '/lossHistory.csv', index_col = None)
    lossHistory = df['loss']
    lrHistory   = df['lr']

    "-------------------------------------------------------------------------"
    df = pd.DataFrame(data = aRNNbefore, columns = cols)
    df.to_csv(jmdDir + "/aRNNbefore.csv", index  = False)
    
    df = pd.DataFrame(data = aRNNafter, columns = cols)
    df.to_csv(jmdDir + "/aRNNafter.csv", index  = False)

    "-------------------------------------------------------------------------"
    yLB = 0.0
    yUB = 0.06
    aMax = 0.05

    nFlightsPerDay = 4
    N     = np.arange(0,nCycles,1)/nFlightsPerDay/365
    nYears = nCycles/nFlightsPerDay/365
    
    "-------------------------------------------------------------------------"
    matplotlib.rc('font', size=14)
    ifig = 0

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[2], colors[0], colors[1], colors[3]]
    color_gray = [0.5, 0.5, 0.5]


    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()

    plt.plot(N, SFleet[:,idxTrain[-1]], color = colors[3], label = 'most agressive mission')
    plt.plot(N, SFleet[:,idxTrain[0]], color = colors[0], label = 'most mild mission')
    plt.xlabel('years')
    plt.ylabel('far-field stress range, $\Delta S$ (MPa)')
#    plt.xlim(0,nYears+0.05)
#    plt.ylim(0,aMax)
    plt.legend(loc = 'upper left',facecolor = 'w')
    plt.grid(which = 'both')

    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()

    aObs = aFleet[-1, idxTrain]
    plt.plot(N, aFleet, color = color_gray)
    plt.plot(N, aFleet[:,idxTrain[-1]], color = colors[3], linewidth = 3, label = 'most agressive mission')
    plt.plot(N, aFleet[:,idxTrain[0]], color = colors[0], linewidth = 3, label = 'most mild mission')
    plt.plot(N, aFleet[:,0], color = color_gray, label = 'fleet cumulative damage history')
    plt.plot(N[-1]*np.ones(aObs.shape), aObs, 'ok', label = 'observed data')
    plt.xlabel('years')
    plt.ylabel('crack length, a (m)')
    plt.xlim(0,nYears+0.05)
    plt.ylim(0,aMax)
    plt.legend(loc = 'upper left',facecolor = 'w')
    plt.grid(which = 'both')
#    
#    "-------------------------------------------------------------------------"
#    ifig = ifig + 1
#    fig = plt.figure(ifig)
#    fig.clf()
#    
#    plt.hist(aFleet[-1, :], bins = np.linspace(0,0.05,11), color = color_gray, label = 'fleet')
#    plt.hist(aFleet[-1, idxTrain], bins = np.linspace(0,0.05,11), color = 'k', label = 'observed data')
#    plt.grid(which = 'both')
#    plt.legend(loc = 'upper right',facecolor = 'w')
#    
    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()

    nEpoch = len(lossHistory)+1
    plt.plot(np.arange(1,nEpoch,1), lossHistory, color = 'k')
    plt.xlim(0,nEpoch)
    plt.xlabel('epoch')
    plt.ylabel('mean square error')
    plt.yscale('log')
    plt.grid(which='both')
#    fig.savefig('lossHistory.png')

##    "-------------------------------------------------------------------------"
##    ifig = ifig + 1
##    fig = plt.figure(ifig)
##    fig.clf()
##
##    plt.plot(lrHistory)
##    plt.xlabel('epoch')
##    plt.ylabel('learning rate')
##    plt.yscale('log')
##    plt.grid(which='both')
#    
    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()
    
    strBefore = "before training\nMSE = %1.1e\nMAE = %1.1e" % (mseBefore, maeBefore)
    strAfter  = "after training\nMSE = %1.1e\nMAE = %1.1e" % (mseAfter, maeAfter)
    
    plt.plot([yLB, yUB], [yLB, yUB], '--k')
    plt.plot(aFleet[-1, idxPredict], aRNNbefore[-1,:], 'o', label = strBefore)
    plt.plot(aFleet[-1, idxPredict], aRNNafter[-1,:], 'o', label = strAfter)
    plt.xlabel('actual')
    plt.ylabel('predicted')
#    plt.xlim(yLB, yUB)
#    plt.ylim(yLB, yUB)
    plt.grid(which='both')
#    plt.axis('square')
    plt.legend(loc = 'lower right',facecolor = 'w')

#    "-------------------------------------------------------------------------"
#    ifig = ifig + 1
#    fig = plt.figure(ifig)
#    fig.clf()
#    
#    plt.plot([yLB, yUB], [yLB, yUB], '--k')
#    plt.plot(aFleet[-1, idxPredict], aRNNbefore[-1,:], 'o')
#    plt.plot(aFleet[-1, idxPredict], aRNNafter[-1,:], 'o')
#    plt.xlabel('actual')
#    plt.ylabel('predicted')
##    plt.xlim(yLB, yUB)
##    plt.ylim(yLB, yUB)
#    plt.grid(which='both')
#    plt.axis('square')
#    
#
#
#    "-------------------------------------------------------------------------"
#    ifig = ifig + 1
#    fig = plt.figure(ifig)
#    fig.clf()
#
#    plt.plot(N, aRNNbefore, color = [0.5, 0.5, 0.5])
#    plt.xlabel('years')
#    plt.ylabel('crack length, a (m)')
#    plt.xlim(0,nYears)
#    plt.ylim(0,aMax)
#    plt.grid(which = 'both')
#
#    "-------------------------------------------------------------------------"
#    ifig = ifig + 1
#    fig = plt.figure(ifig)
#    fig.clf()
#
#    plt.plot(N, aRNNafter, color = [0.5, 0.5, 0.5])
#    plt.xlabel('years')
#    plt.ylabel('crack length, a (m)')
#    plt.xlim(0,nYears)
#    plt.ylim(0,aMax)
#    plt.grid(which = 'both')
    
    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()

    aObs = aFleet[-1, idxTrain]
    plt.plot(N, np.divide(aRNNbefore, aFleet[:, idxPredict]), color = colors[1])
    plt.xlabel('years')
    plt.ylabel('$a_{predicted}$ / $a_{actual}$')
    plt.xlim(0,nYears+0.05)
    plt.ylim(0,2.5)
    plt.grid(which = 'both')

    "-------------------------------------------------------------------------"
    ifig = ifig + 1
    fig = plt.figure(ifig)
    fig.clf()

    aObs = aFleet[-1, idxTrain]
    plt.plot(N, np.divide(aRNNafter, aFleet[:, idxPredict]), color = colors[2])
    plt.xlabel('years')
    plt.ylabel('$a_{predicted}$ / $a_{actual}$')
    plt.xlim(0,nYears+0.05)
    plt.ylim(0,2.5)
    plt.grid(which = 'both')
