import tensorflow as tf
import numpy as np


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
