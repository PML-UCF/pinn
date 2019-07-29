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
import pandas as pd

import matplotlib.pyplot as plt

from model import create_model

# --------------------------------------------------------------------------
if __name__ == "__main__":

    myDtype = 'float32'

    # --------------------------------------------------------------------------
    # Sequence for importing input loads for each asset as time series
    df = pd.read_csv('Loads_1000cycles.csv', index_col=None, dtype=myDtype)
    macArray = np.transpose(np.asarray(df))
    nFleet, nCycles = macArray.shape
    avgLoads = np.linspace(100, 500, nFleet, dtype=myDtype)

    # --------------------------------------------------------------------------
    # Preliminary parameters

    # The spherical bearing life equation used in this sample case is given in the reference:
    # "Y. A. Yucesan and F. A. C. Viana, ''Onshore wind turbine main bearing 
    # reliability and its implications in fleet management,'' in AIAA
    # SciTech Forum, (San Diego, USA), pp. AIAA 2019--1225, AIAA, 2019."

    # The equation is N = a1*askf*(C/P)**(10/3)
    # Linearization of this equation in the log10-log10 space gives below linear equation parameters as:
    # a = -10/3, b = log10(a1) + log10(askf) + 10/3*log10(C)
    #    where:
    #        * a1 is the life modification factor for reliability level (taken as 1 for 10% unreliability)
    #        * askf is the life modification factor related to grease parameters (taken as 1 for simplicity)
    #        * C is the basic dynamic load rating (taken as 6000 kN for SKF 230/600 CAW33 model bearing)
    #        * P is the equivalent dynamic bearing load (cyclic input load)
    #        * Life cycles in millions for corresponding P

    a = -10 / 3  # Slope of linearized SN-Curve in log10-log10 space
    b = 12.594  # Interception of linearized SN-Curve in log10-log10 space
    N = a * np.log10(macArray) + b  # Number of cycles for corresponding load
    d = 1 / 10 ** N  # Delta damage
    d0 = 0.  # Initial damage
    ndex = [1]  # To filter load from inputs
    # --------------------------------------------------------------------------
    # Sequence for damage history calculation
    dHistAll = []
    for mac in d:
        dmgCum = 0
        dHist = []
        for dCyc in mac:
            dmgCum += dCyc
            dHist.append(dmgCum)
        dHistAll.append(dHist)
    dHistAll = np.asarray(dHistAll, dtype=myDtype)

    # Input loads tensor manipulations
    Sobs = macArray[:, :, np.newaxis]
    Sobs = np.log10(Sobs)
    batch_input_shape = Sobs.shape

    # --------------------------------------------------------------------------
    # Prediction sequence
    da0RNN = d0 * np.ones((Sobs.shape[0], 1), dtype=myDtype)

    model = create_model(a=a, b=b, batch_input_shape=batch_input_shape, ndex=ndex,
                         da0RNN=da0RNN, myDtype=myDtype, return_sequences=True)

    results = model.predict_on_batch(Sobs)[:, :, 0]
    # --------------------------------------------------------------------------
    # Plot damage history for all machines
    fig = plt.figure(1)
    fig.clf()

    plt.plot(np.transpose(np.repeat([range(nCycles)], nFleet, axis=0)), np.transpose(results))

    plt.title('Damage History')
    plt.xlabel('Million Cycles')
    plt.ylabel('Damage')
    plt.grid(which='both')
    plt.show()

    # Plot SN-Curve
    fig = plt.figure(2)
    fig.clf()

    plt.plot(1 / dHistAll[:, -1], avgLoads, 'b-', label='SN model')
    plt.plot(1 / results[:, -1], avgLoads, 'r--', label='PINN')

    plt.title('SN Curve')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Million Cycles')
    plt.ylabel('Load (kN)')
    plt.grid(which='both')
    plt.legend(loc=0, facecolor='w')
    plt.show()
