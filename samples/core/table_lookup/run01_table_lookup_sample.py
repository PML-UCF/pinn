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

""" Table lookup interpolation sample case
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.python.framework import ops

import sys
sys.path.append('../../../')

from pinn.layers.core import tableInterpolation

# Model
def create_model(grid_array, bounds, input_shape, table_shape):
    dLInterpol = tableInterpolation(input_shape = input_shape)
    dLInterpol.build(input_shape = table_shape)
    dLInterpol.set_weights([grid_array, bounds])
    model = tf.keras.Sequential()
    model.add(dLInterpol)
    return model

# Input Table
myDtype = tf.float32
#df = pd.read_csv('aSKF_kappa1.csv') # 1 Dimensional Table (f(x) = y)
df = pd.read_csv('aSKF_kappa12.csv') # 2 Dimensional Table (f(x1,x2) = y)
data = np.transpose(np.asarray(np.transpose(df))[1:])
if data.shape[1] == 1:
    data = np.repeat(data,2,axis=1)
space = np.asarray([np.asarray(df['xval']),np.asarray([float(i) for i in df.columns[1:]])])
table_shape = (data.shape[0],2)
bounds = np.asarray([[np.min(space[0]),np.max(space[0])],[np.min(space[1]),np.max(space[1])]])

# Input Query Points
q = np.asarray([[0.05,1.5],[0.1,1.0],[0.3,2.5],[2.0,2.0]])
input_array = ops.convert_to_tensor(q,dtype=tf.float32)
input_shape = input_array.shape
input_array= tf.expand_dims(input_array,0)

model = create_model(data, bounds, input_shape, table_shape)
result = model.predict(input_array, steps = 1)
print(result)
