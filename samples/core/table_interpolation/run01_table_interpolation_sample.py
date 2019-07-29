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

from pinn.layers import TableInterpolation

if __name__ == "__main__":
    
    # Model
    def create_model(grid_array, bounds, input_shape, table_shape):
        
        model = tf.keras.Sequential()
        dLInterpol = TableInterpolation(table_shape = table_shape)
        dLInterpol.build(input_shape)
        dLInterpol.set_weights([grid_array, bounds])
        model.add(dLInterpol)
               
        return model
    
    # Input table manipulation
    #df = pd.read_csv('aSKF_kappa1.csv') # 1 Dimensional Table (f(x) = y)
    df = pd.read_csv('aSKF.csv') # 2 Dimensional Table (f(x1,x2) = y)
    
    table = np.transpose(np.asarray(np.transpose(df))[1:])

    if table.shape[1] == 1:
        table = np.repeat(table,2,axis=1) # This line converts the table into a 2D format if it is 1D
    # Table shape should be in the form of (1,table.shape[0],table.shape[1],1) to comply with the class
    table = np.expand_dims(table,0)
    table = np.expand_dims(table,-1)
    # Fetch the upper and lower bounds of the axes of the table
    grid_space = np.asarray([np.asarray(df['xval']),np.asarray([float(i) for i in df.columns[1:]])])
    bounds = np.asarray([[np.min(grid_space[0]),np.min(grid_space[1])],[np.max(grid_space[0]),np.max(grid_space[1])]])
    table_shape = table.shape
    
    # Input query points
    query_points = np.asarray([[2.0,1.0],[1.0,0.9],[0.3,0.5]])
    query_points = query_points[np.newaxis,:,:]
    input_shape = query_points.shape
    
    # Build the model and pass the query points
    model = create_model(table, bounds, input_shape, table_shape)
    result = model.predict(query_points)
    print(result)
