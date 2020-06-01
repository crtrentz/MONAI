# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from skimage import io, transform


def write_png(
    data,
    file_name,
    output_shape=None,
    interp_order=3,
    mode="constant",
    cval=0,
    scale=False,
    plugin=None,
    **plugin_args,
):
    """
    Write numpy data into png files to disk.  
    Spatially It supports HW for 2D.(H,W) or (H,W,3) or (H,W,4)
    It's based on skimage library: https://scikit-image.org/docs/dev/api/skimage

    Args:
        data (numpy.ndarray): input data to write to file.
        file_name (string): expected file name that saved on disk.
        output_shape (None or tuple of ints): output image shape.
        interp_order (int): the order of the spline interpolation, default is InterpolationCode.SPLINE3.
            The order has to be in the range 0 - 5.
            this option is used when `output_shape != None`.
        mode (`reflect|constant|nearest|mirror|wrap`):
            The mode parameter determines how the input array is extended beyond its boundaries.
            this option is used when `output_shape != None`.
        cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
            this option is used when `output_shape != None`.
        scale (bool): whether to scale data with 255 and convert to uint8 for data in range [0, 1].
        plugin (string): name of plugin to use in `imsave`. By default, the different plugins
            are tried(starting with imageio) until a suitable candidate is found.
        plugin_args (keywords): arguments passed to the given plugin.

    """
    assert isinstance(data, np.ndarray), "input data must be numpy array."

    if output_shape is not None:
        assert (
            isinstance(output_shape, (list, tuple)) and len(output_shape) == 2
        ), "output_shape must be a list of 2 values (H, W)."

        if len(data.shape) == 3:
            output_shape += (data.shape[2],)

        data = transform.resize(data, output_shape, order=interp_order, mode=mode, cval=cval, preserve_range=True)

    if scale:
        assert np.min(data) >= 0 and np.max(data) <= 1, "png writer only can scale data in range [0, 1]."
        data = 255 * data
    data = data.astype(np.uint8)
    io.imsave(file_name, data, plugin=plugin, **plugin_args)
    return
