# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:58:47 2019

@author: ar679403
"""

import numpy as np
import matplotlib.pyplot as plt

def threshold(dai,dap,a,ath):
    alpha = 1e6
    m = 1/(1+np.exp(-alpha*(a-ath)))
    da = m*dap+(1-m)*dai
    return da
#--------------------------------------------------------------------------
if __name__ == "__main__":
    
    ath = .5e-3
    a = np.linspace(0,1e-3,1000)
    alpha = 1e6
    sig = 1/(1+np.exp(-alpha*(a-ath)))
    
    dai = .25e-3*np.ones(len(a))
    dap = .75e-3*np.ones(len(a))
    da = threshold(dai,dap,a,ath)
    #--------------------------------------------------------------------------
    fig  = plt.figure(1)
    fig.clf()
    
    plt.plot(a*1e3,sig)
    
    plt.title('Sigmoid calibration: alpha = '+str(int(alpha)))
    plt.xlabel('crack length [mm]')
    plt.grid(which = 'both')
    
    fig  = plt.figure(2)
    fig.clf()
    
    plt.plot(a*1e3,da*1e3,'-k')
    plt.plot(a*1e3,dai*1e3,':r', label = 'init.')
    plt.plot(a*1e3,dap*1e3,':b', label = 'prop.')
    
    plt.title('Threshold response (numpy function)')
    plt.xlabel('crack length [mm]')
    plt.ylabel('$\Delta$ a [mm]')
    plt.legend(loc=0, facecolor = 'w')
    plt.grid(which = 'both')
    #--------------------------------------------------------------------------
    
