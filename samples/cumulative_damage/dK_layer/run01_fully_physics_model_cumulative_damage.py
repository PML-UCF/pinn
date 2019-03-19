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
from tensorflow.python.framework import ops

import numpy as np
import pandas as pd


import sys
sys.path.append('../../../')

from model import create_physics_model

#--------------------------------------------------------------------------
if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
    # preliminaries
    myDtype = tf.float32

    a0   = 0.005       # initial crack length [m]
    m    = 3.8         # Paris model exponent
    C    = 1.5E-11     # Paris model constant
    F    = 1.0

    #--------------------------------------------------------------------------
    # fleet information
    df = pd.read_csv('aFleet_5yrs.csv', index_col = None)
    aFleet = np.asarray(df)
    
    df = pd.read_csv('SFleet_5yrs.csv', index_col = None)
    SFleet = np.transpose(np.asarray(df))
    nFleet, nCycles = SFleet.shape
    
    #--------------------------------------------------------------------------
    SFleet = SFleet[:,:,np.newaxis]
    SFleet = ops.convert_to_tensor(SFleet, dtype = myDtype)
    
    batch_input_shape = SFleet.shape

    a0RNN = ops.convert_to_tensor(a0 * np.ones((nFleet, 1)), dtype=myDtype)
    
    #--------------------------------------------------------------------------
    modelPhysics = create_physics_model(F = F, C = C, m = m, inputs = SFleet,
                         batch_input_shape = batch_input_shape, a0RNN = a0RNN,
                         myDtype = myDtype, return_sequences = True)
    
    aPhysics = modelPhysics.predict_on_batch(SFleet)[:,:,0].transpose()