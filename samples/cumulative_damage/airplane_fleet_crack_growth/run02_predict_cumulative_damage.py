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

""" Predict with physics-informed recursive neural network
"""

import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as matplotlib

from model import create_model

# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # preliminaries
    myDtype = 'float32'

    a0 = 0.005  # initial crack length [m]
    m = 3.8  # Paris model exponent
    C = 1.5E-11  # Paris model constant

    # --------------------------------------------------------------------------
    # fleet information
    df = pd.read_csv('aFleet_5yrs.csv', index_col=None, dtype=myDtype)
    aFleet = np.asarray(df)

    df = pd.read_csv('SFleet_5yrs.csv', index_col=None, dtype=myDtype)
    SFleet = np.transpose(np.asarray(df))
    nFleet, nCycles = SFleet.shape

    # --------------------------------------------------------------------------
    SFleet = SFleet[:, :, np.newaxis]

    batch_input_shape = SFleet.shape

    a0RNN = a0 * np.ones((nFleet, 1), dtype=myDtype)

    # --------------------------------------------------------------------------
    dkLayer = tf.keras.models.load_model('DK_MLP.h5')
    dkLayer.trainable = True

    modelBefore = create_model(dkLayer=dkLayer, C=C, m=m,
                               batch_input_shape=batch_input_shape, a0RNN=a0RNN,
                               myDtype=myDtype, return_sequences=True)

    aRNNbefore = modelBefore.predict_on_batch(SFleet)[:, :, 0].transpose()
    errBefore = aFleet[-1, :] - aRNNbefore[-1, :]
    mseBefore = np.mean(errBefore ** 2.0)
    maeBefore = np.max(np.abs(errBefore))

    model_dir = "./training_60_points"
    weight_path = model_dir + "/cp.ckpt"

    modelAfter = create_model(dkLayer=dkLayer, C=C, m=m,
                              batch_input_shape=batch_input_shape, a0RNN=a0RNN,
                              myDtype=myDtype, return_sequences=True)
    modelAfter.load_weights(weight_path)

    aRNNafter = modelAfter.predict_on_batch(SFleet)[:, :, 0].transpose()
    errAfter = aFleet[-1, :] - aRNNafter[-1, :]
    mseAfter = np.mean(errAfter ** 2.0)
    maeAfter = np.max(np.abs(errAfter))

    # --------------------------------------------------------------------------
    matplotlib.rc('font', size=14)

    yLB = 0.0
    yUB = 0.06

    # --------------------------------------------------------------------------
    ifig = 1
    fig = plt.figure(ifig)
    fig.clf()

    strBefore = "before training\nMSE = %1.1e\nMAE = %1.1e" % (mseBefore, maeBefore)
    strAfter = "after training\nMSE = %1.1e\nMAE = %1.1e" % (mseAfter, maeAfter)

    plt.plot([yLB, yUB], [yLB, yUB], '--k')
    plt.plot(aFleet[-1, :], aRNNbefore[-1, :], 'o', label=strBefore)
    plt.plot(aFleet[-1, :], aRNNafter[-1, :], 'o', label=strAfter)
    plt.xlabel('actual crack length (m)')
    plt.ylabel('predicted crack length (m)')
    plt.xlim(yLB, yUB)
    plt.ylim(yLB, yUB)
    plt.grid(which='both')
    plt.legend(loc='lower right', facecolor='w')
    plt.show()
