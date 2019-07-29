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
""" Utility functions
"""

#from tensorflow.contrib.image.python.ops.dense_image_warp import _interpolate_bilinear as interpolate
#from tensorflow_addons.image.dense_image_warp import interpolate_bilinear as interpolate

from tensorflow import reshape, shape, expand_dims, convert_to_tensor
from tensorflow import unstack, cast, gather
from tensorflow import constant
from tensorflow import name_scope, control_dependencies

from tensorflow import range as tfrange

from tensorflow.debugging import assert_equal, assert_greater_equal, assert_less_equal

from tensorflow.math import minimum, maximum, floor

from tensorflow.dtypes import int32, float32

import numpy as np


def interpolate(grid, query_points, indexing="ij", name=None):
    """
    Reference: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/image/dense_image_warp.py
    
    Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).
    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be \'ij\' or \'xy\'")

    with name_scope(name or "interpolate_bilinear"):
        grid = convert_to_tensor(grid)
        query_points = convert_to_tensor(query_points)

        if len(grid.shape) != 4:
            msg = "Grid must be 4 dimensional. Received size: "
            raise ValueError(msg + str(grid.shape))

        if len(query_points.shape) != 3:
            raise ValueError("Query points must be 3 dimensional.")

        if query_points.shape[2] is not None and query_points.shape[2] != 2:
            raise ValueError("Query points must be size 2 in dim 2.")

        if grid.shape[1] is not None and grid.shape[1] < 2:
            raise ValueError("Grid height must be at least 2.")

        if grid.shape[2] is not None and grid.shape[2] < 2:
            raise ValueError("Grid width must be at least 2.")

        grid_shape = shape(grid)
        query_shape = shape(query_points)

        batch_size, height, width, channels = (grid_shape[0], grid_shape[1],
                                               grid_shape[2], grid_shape[3])

        shape_list = [batch_size, height, width, channels]

        # pylint: disable=bad-continuation
        with control_dependencies([
                assert_equal(
                    query_shape[2],
                    2,
                    message="Query points must be size 2 in dim 2.")
        ]):
            num_queries = query_shape[1]
        # pylint: enable=bad-continuation

        query_type = query_points.dtype
        grid_type = grid.dtype

        # pylint: disable=bad-continuation
        with control_dependencies([
                assert_greater_equal(
                    height, 2, message="Grid height must be at least 2."),
                assert_greater_equal(
                    width, 2, message="Grid width must be at least 2."),
        ]):
            alphas = []
            floors = []
            ceils = []
            index_order = [0, 1] if indexing == "ij" else [1, 0]
            unstacked_query_points = unstack(query_points, axis=2)
        # pylint: enable=bad-continuation

        for dim in index_order:
            with name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape_list[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant(0.0, dtype=query_type)
                floor_val = minimum(
                    maximum(min_floor, floor(queries)),
                    max_floor)
                int_floor = cast(floor_val, int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = cast(queries - floor_val, grid_type)
                min_alpha = constant(0.0, dtype=grid_type)
                max_alpha = constant(1.0, dtype=grid_type)
                alpha = minimum(
                    maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = expand_dims(alpha, 2)
                alphas.append(alpha)

        # pylint: disable=bad-continuation
        with control_dependencies([
                assert_less_equal(
                    cast(
                        batch_size * height * width, dtype=float32),
                    np.iinfo(np.int32).max / 8.0,
                    message="The image size or batch size is sufficiently "
                    "large that the linearized addresses used by tf.gather "
                    "may exceed the int32 limit.")
        ]):
            flattened_grid = reshape(
                grid, [batch_size * height * width, channels])
            batch_offsets = reshape(
                tfrange(batch_size) * height * width, [batch_size, 1])
        # pylint: enable=bad-continuation

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather_fn(y_coords, x_coords, name):
            with name_scope("gather-" + name):
                linear_coordinates = (
                    batch_offsets + y_coords * width + x_coords)
                gathered_values = gather(flattened_grid, linear_coordinates)
                return reshape(gathered_values,
                                  [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather_fn(floors[0], floors[1], "top_left")
        top_right = gather_fn(floors[0], ceils[1], "top_right")
        bottom_left = gather_fn(ceils[0], floors[1], "bottom_left")
        bottom_right = gather_fn(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (
                bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


