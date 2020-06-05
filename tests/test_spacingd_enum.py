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

import unittest

import numpy as np

from monai.transforms import Spacingd
from monai.utils.enums import InterpolationOrder, ExtendMode


class TestSpacingDEnum(unittest.TestCase):
    def test_interp_all(self):
        data = {
            "image": np.arange(20).reshape((2, 10)),
            "seg": np.ones((2, 10)),
            "image.affine": np.eye(4),
            "seg.affine": np.eye(4),
        }
        spacing = Spacingd(
            keys=("image", "seg"), interp_order=InterpolationOrder.SPLINE0, pixdim=(0.2,), mode=ExtendMode.NEAREST
        )
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))

    def test_interp_sep_tuple_enum(self):
        data = {"image": np.ones((2, 10)), "seg": np.ones((2, 10)), "image.affine": np.eye(4), "seg.affine": np.eye(4)}
        spacing = Spacingd(
            keys=("image", "seg"),
            interp_order=(InterpolationOrder.SPLINE2, InterpolationOrder.SPLINE0),
            pixdim=(0.2,),
            mode=ExtendMode.NEAREST,
        )
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))

    def test_interp_sep_tuple_mix(self):
        data = {"image": np.ones((2, 10)), "seg": np.ones((2, 10)), "image.affine": np.eye(4), "seg.affine": np.eye(4)}
        spacing = Spacingd(
            keys=("image", "seg"), interp_order=(2, InterpolationOrder.SPLINE0), pixdim=(0.2,), mode=ExtendMode.NEAREST
        )
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))

    def test_interp_sep_list_enum(self):
        data = {"image": np.ones((2, 10)), "seg": np.ones((2, 10)), "image.affine": np.eye(4), "seg.affine": np.eye(4)}
        spacing = Spacingd(
            keys=("image", "seg"),
            interp_order=[InterpolationOrder.SPLINE2, InterpolationOrder.SPLINE0],
            pixdim=(0.2,),
            mode=ExtendMode.NEAREST,
        )
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))

    def test_interp_sep_list_mix(self):
        data = {"image": np.ones((2, 10)), "seg": np.ones((2, 10)), "image.affine": np.eye(4), "seg.affine": np.eye(4)}
        spacing = Spacingd(
            keys=("image", "seg"), interp_order=[2, InterpolationOrder.SPLINE0], pixdim=(0.2,), mode=ExtendMode.NEAREST
        )
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))

    def test_interp_sep_list_int(self):
        data = {"image": np.ones((2, 10)), "seg": np.ones((2, 10)), "image.affine": np.eye(4), "seg.affine": np.eye(4)}
        spacing = Spacingd(keys=("image", "seg"), interp_order=[2, 0], pixdim=(0.2,), mode=ExtendMode.NEAREST)
        res = spacing(data)
        self.assertEqual(("image", "image.affine", "seg", "seg.affine"), tuple(sorted(res)))
        np.testing.assert_allclose(res["image"].shape, (2, 46))
        np.testing.assert_allclose(res["image.affine"], np.diag((0.2, 1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
