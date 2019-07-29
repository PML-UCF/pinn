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

import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import simple_rnn_model
import gru_model
import lstm_model

import cd_rnn_model


from tensorflow.keras import optimizers, constraints, initializers, Sequential, losses
from tensorflow.python.keras.layers import LSTM_v2 as LSTM, GRU_v2 as GRU, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


import train_func

if __name__ == "__main__":
    K.clear_session()
    # --------------------------------------------------------------------------
    # preliminaries
    myDtype = 'float32'

    a0 = 0.005  # initial crack length [m]
    m = 3.8  # Paris model exponent
    C = 1.5E-11  # Paris model constant

    # --------------------------------------------------------------------------
    # fleet information
    df = pd.read_csv('aFleet_5yrs.csv', index_col=None)
    aFleet = np.transpose(np.asarray(df))

    df = pd.read_csv('SFleet_5yrs.csv', index_col=None)
    SFleet = np.transpose(np.asarray(df))
    nFleet, nCycles = SFleet.shape

    # --------------------------------------------------------------------------
    idx = np.argsort(aFleet[:, -1])

    arange = np.linspace(0, 299, 60, dtype=int)
    idxTrain = idx[arange]

    Sobs = SFleet[idxTrain, :]
    Sobs = Sobs[:, :, np.newaxis]



    SFleet = SFleet[:, :, np.newaxis]

    nObs = Sobs.shape[0]

    # --------------------------------------------------------------------------
    aTarget = aFleet[idxTrain, :]
    aTarget = aTarget[:, :, np.newaxis]

    aFleet = aFleet[:, :, np.newaxis]
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # LSTM model
    # --------------------------------------------------------------------------
    rnn_type = 'gru'
    layers = 4
    units = 32
    epochs = 1000
    n_timestep = 1000

    idx = np.arange(0, 7300, 7300/n_timestep, dtype=int)
    # idx = [0, 7299]

    training_path = './training/{}_{}x{}_{}'.format(rnn_type, layers, units, n_timestep)

    xYears = np.arange(0, nCycles) / 4. / 365.
    xTimeStamp = xYears[idx]

    x = Sobs[:, idx, :]
    y = aTarget[:, idx, :]

    x_val = SFleet[:, idx, :]
    y_val = aFleet[:, idx, :]

    x_scale = train_func.Scale((-1, 1))
    y_scale = train_func.Scale((0, 1))

    x_val_scale = train_func.Scale((-1, 1))
    y_val_scale = train_func.Scale((0, 1))

    # shape: [timestep, samples, feature]
    # x = Sobs[:, :, 0].transpose()[idx, :, np.newaxis]
    # y = aTarget[:, :, 0].transpose()[idx, :, np.newaxis]
    #
    # x_val = SFleet[:, :, 0].transpose()[idx, :, np.newaxis]
    # y_val = aFleet[:, :, 0].transpose()[idx, :, np.newaxis]

    # fig = plt.figure("Ground Truth")
    # fig.clf()
    # plt.plot(xTimeStamp, y_val[:, :, 0].transpose(), 'gray')
    # plt.plot(np.repeat(xTimeStamp[-1], y_val.shape[0]), y_val[:, -1, 0].transpose(), 'ob')
    # plt.title("Ground Truth")
    # plt.show()

    batch_input_shape = x.shape


    def layer_mse(y_true, y_pred, layer=-1):
        pred = y_pred[:, layer, 0]
        true = y_true[:, layer, 0]

        pred = ops.convert_to_tensor(pred)
        true = math_ops.cast(true, pred.dtype)
        return K.mean(math_ops.squared_difference(pred, true), axis=-1)


    def last_mse(y_true, y_pred):
        return layer_mse(y_true, y_pred, layer=-1)


    def first_mse(y_true, y_pred):
        return layer_mse(y_true, y_pred, layer=0)


    def create_model(
            units=1, layers=1, batch_input_shape=(None,), initial_state=0.0, stateful=False, return_sequences=False,
            idx_mask=None, dense_units=1, da_max=5e-5, da_min=5e-5
    ):
        model = Sequential()

        # lstm = LSTM(units, batch_input_shape=batch_input_shape, stateful=stateful, return_sequences=return_sequences,
        #             kernel_initializer='zeros', recurrent_initializer='glorot_uniform',
        #             recurrent_constraint='non_neg', kernel_constraint=None,
        #             activation='elu', recurrent_activation='elu', unit_forget_bias=False
        #            )
        for i in range(layers):
            lstm = GRU(
                int(units), batch_input_shape=batch_input_shape, stateful=stateful, return_sequences=return_sequences,
                kernel_initializer=initializers.Constant(1e-3)
            )
            model.add(lstm)

        # model.add(Dense(units, activation='tanh', kernel_initializer=initializers.Constant(1e-2)))

        if dense_units is not None and units > 1:
            # model.add(Dense(1, kernel_initializer='glorot_uniform', activation='elu', kernel_constraint=None))
            model.add(Dense(dense_units, kernel_initializer='glorot_normal', use_bias=False))

        # set initial states (2 states for LSTM) - initial state for h and c
        # lstm.reset_states([
        #     initial_state * np.ones((batch_input_shape[0], units)),
        #     initial_state * np.ones((batch_input_shape[0], units))
        # ])

        model.layers[0](
            tf.zeros(batch_input_shape),
            initial_state=[
                initial_state * tf.ones((batch_input_shape[0], units))
            ]
        )

        mse = losses.MeanSquaredError()
        loss = lambda y_true, y_pred:  mse(y_true, y_pred)*1e1 + last_mse(y_true, y_pred)
        # loss = lambda y_true, y_pred:  mse(y_true, y_pred) + last_mse(y_true, y_pred)*1e-1 + first_mse(y_true, y_pred)*1e3

        # model.compile(loss=loss_fn(idx_mask), optimizer=optimizers.Adam(1e-4))
        model.compile(loss=loss, optimizer=optimizers.Adam(1e-3), metrics=[mse, first_mse, last_mse])

        model.idx_mask = idx_mask

        return model


    def create_fn(batch_input_shape):
        model = create_model(
            units=units, layers=layers, dense_units=1, batch_input_shape=batch_input_shape, initial_state=a0,
            stateful=False, return_sequences=True, idx_mask=None
        )

        # path = './training/lstm_8x32_1000'
        # model.load_weights(path + '/cp.ckpt')
        #
        # w = model.get_weights()
        #
        # model = create_model(
        #     units=units, layers=layers, dense_units=1, batch_input_shape=batch_input_shape, initial_state=a0,
        #     stateful=False, return_sequences=True, idx_mask=None
        # )
        #
        # w[-1] = w[-1]/2
        #
        # model.set_weights(w)

        return model

    train_func.train(
        create_fn, x, y,
        training_path=training_path, epochs=epochs, reduce_lr=20, early_stop=200, x_scale=x_scale, y_scale=y_scale
    )

    train_func.evaluate(create_fn, x_val, y_val, training_path=training_path, x_scale=x_scale, y_scale=y_scale)

    m = create_model(
        units=units, layers=layers, batch_input_shape=x_val.shape, initial_state=a0, return_sequences=True, stateful=False
    )
    m.load_weights(training_path+'/cp.ckpt')

