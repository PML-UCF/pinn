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

import numpy as np
import pandas as pd

from model import create_model

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
    idx = np.argsort(aFleet[-1, :])

    arange = np.asarray(np.linspace(0, 299, 60), dtype=int)
    idxTrain = idx[arange]

    Sobs = SFleet[idxTrain, :]
    Sobs = Sobs[:, :, np.newaxis]

    batch_input_shape = Sobs.shape

    SFleet = SFleet[:, :, np.newaxis]

    nObs = Sobs.shape[0]

    # --------------------------------------------------------------------------
    aTarget = aFleet[-1, idxTrain]
    aTarget = aTarget[:, np.newaxis]

    a0RNN = a0 * np.ones((nObs, 1), dtype=myDtype)

    # --------------------------------------------------------------------------
    dkLayer = tf.keras.models.load_model('DK_MLP.h5')
    dkLayer.trainable = True

    model = create_model(dkLayer=dkLayer, C=C, m=m,
                         batch_input_shape=batch_input_shape, a0RNN=a0RNN, myDtype=myDtype)

    # --------------------------------------------------------------------------
    EPOCHS = 5
    jmdDir = "./training_%d_points" % len(idxTrain)

    weight_path = jmdDir + "/cp.ckpt"
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=weight_path, monitor='loss',
                                                         verbose=1, save_best_only=True,
                                                         mode='min', save_weights_only=True)

    callbacks_list = [ModelCheckpoint]

    history = model.fit(Sobs, aTarget, epochs=EPOCHS, steps_per_epoch=1, verbose=1, callbacks=callbacks_list)

    # --------------------------------------------------------------------------
    df = pd.DataFrame.from_dict(history.history)
    df.insert(loc=0, column='epoch', value=history.epoch)
    df.to_csv(jmdDir + "/lossHistory.csv", index=False)
