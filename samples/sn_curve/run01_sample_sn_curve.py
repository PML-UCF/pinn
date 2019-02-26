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

""" SN-Curve sample with RNN structure
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

import sys
sys.path.append('../../')
from SNblock_model import create_model


myDtype = tf.float32
cycles = 1000                          # Number of cycles
machines = 50                          # Number of machines
loads = np.linspace(100,1000,machines) # Average loads per machine

# Sequence for load time series generation randomly around average loads
import random
maclist = []
for m in range(machines):
    loadhist = []
    for c in range(cycles):
        loadhist.append(loads[m]*(random.random()+0.5))
    maclist.append(loadhist)
macarray = np.asarray(maclist)

a = -10/3                  # Slope of linearized SN-Curve in log10-log10 space
b = 13.372                 # Interception of linearized SN-Curve in log10-log10 space
N = a*np.log10(macarray)+b # Number of cycles for corresponding load level
d = 1/10**N                # Delta damage
do = 0                     # Initial damage


# Sequence for damage history calculation
dhistall = []
for mac in d:
    dmgcum = 0
    dhist = []
    for dcyc in mac:
        dmgcum += dcyc
        dhist.append(dmgcum)
    dhistall.append(dhist)
dhistall = np.asarray(dhistall)

# Input loads tensor manipulations
Sobs = macarray[:, :, np.newaxis]
Sobs = np.log10(Sobs)
Sobs = ops.convert_to_tensor(Sobs, dtype = myDtype)
batch_input_shape = Sobs.shape

# Training and prediction sequence
daTarget = dhistall[:,-1]
daTarget = daTarget[:, np.newaxis] 
daTarget = ops.convert_to_tensor(daTarget, dtype=myDtype)

da0RNN = ops.convert_to_tensor(do * np.ones((Sobs.shape[0], 1)), dtype=myDtype)
    
pre_model = create_model(a = a, b = b, batch_input_shape = batch_input_shape, 
                         da0RNN = da0RNN, myDtype = myDtype)
history = pre_model.fit(Sobs, daTarget ,epochs=5, steps_per_epoch=5)

pre_model.save_weights('model_weights.h5')

model = create_model(a = a, b = b, batch_input_shape = batch_input_shape, 
                     da0RNN = da0RNN, myDtype = myDtype, return_sequences = True)
model.load_weights('model_weights.h5')

results = model.predict_on_batch(Sobs)[:,:,0]
"-------------------------------------------------------------------------"

# Plot damage history for all machines
ifig = 0
ifig = ifig + 1
fig  = plt.figure(ifig)
fig.clf()

plt.plot(np.transpose(np.repeat([range(cycles)],machines,axis=0)), np.transpose(results))

plt.title('Damage History')
plt.xlabel('Cycles')
plt.ylabel('Damage')
plt.grid(which = 'both')

# Plot SN-Curve
fig  = plt.figure(2)
fig.clf()

plt.plot(1/dhistall[:,-1], loads,'b-', label = 'SN model')
plt.plot(1/results[:,-1], loads,'r--', label = 'PINN')

plt.title('SN Curve')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Cycles')
plt.ylabel('Load')
plt.grid(which = 'both')
plt.legend(loc=0, facecolor = 'w')
