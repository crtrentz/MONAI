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

import os
import tempfile
import unittest

from monai.data import write_png, InterpolationCode
from skimage import io, transform
import numpy as np


class TestPngWrite(unittest.TestCase):
    def test_write_gray(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 1)
            img_save_val = 255 * img
            # saving with io.imsave (h, w, 1) will only give us (h,w) while reading it back.
            img_save_val = img_save_val[:, :, 0].astype(np.uint8)
            write_png(img, image_name, scale=True)
            out = io.imread(image_name)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 3, 3)
            img_save_val = (255 * img).astype(np.uint8)
            write_png(img, image_name, scale=True)
            out = io.imread(image_name)
            np.testing.assert_allclose(out, img_save_val)

    def test_write_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.png")
            img = np.random.rand(2, 2, 3)
            write_png(img, image_name, (4, 4), scale=True)
            img_save_val = transform.resize(
                img, (4, 4), order=InterpolationCode.SPLINE3, mode="constant", cval=0, preserve_range=True
            )
            img_save_val = (255 * img_save_val).astype(np.uint8)
            out = io.imread(image_name)
            np.testing.assert_allclose(out, img_save_val)


if __name__ == "__main__":
    unittest.main()
