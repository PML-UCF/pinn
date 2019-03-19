import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.contrib.image.python.ops.dense_image_warp import _interpolate_bilinear as interp
from tensorflow.contrib.resampler import resampler

from tensorflow.python.framework import ops


#    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
#    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
#    name: a name for the operation (optional).
#    indexing: whether the query points are specified as row and column (ij),
#      or Cartesian coordinates (xy).
#
#  Returns:
#    values: a 3-D `Tensor` with shape `[batch, N, channels]`
#
#  Raises:
#    ValueError: if the indexing mode is invalid, or if the shape of the inputs
    

data = np.asarray([[0,0,0],[1,2,3],[10,20,30]])
data = data[np.newaxis,:,:,np.newaxis]
grid = ops.convert_to_tensor(data,dtype=tf.float32)

q = np.asarray([[0.0,0.0],[1.5,1.0],[2.0,2.0]])
q = q[np.newaxis,:,:]
query_points = ops.convert_to_tensor(q,dtype=tf.float32)

out_interp = interp(grid,query_points)
out_resamp = resampler(grid,query_points)

sess = tf.Session()
res_interp = sess.run(out_interp)
res_resamp = sess.run(out_resamp)

print(res_interp)
#print(res_resamp)

xval = np.linspace(0.02,5.0,1000)

askf = 0.91671483*xval**6 - 5.64733622*xval**5 + 13.77524582*xval**4 - 10.90333088*xval**3 + 5.96065039*xval**2 + 0.99228844*xval + 0.15619915

df = pd.DataFrame({'xval' : xval, 'askf' : askf})
df.to_csv('aSKF_kappa1.csv', index = False)

