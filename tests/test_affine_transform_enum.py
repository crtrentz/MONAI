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
import torch
from parameterized import parameterized

from monai.networks.layers import AffineTransform
from monai.networks.utils import normalize_transform, to_norm_affine
from monai.utils.enum import InterpolationMode, PaddingMode


class TestNormTransformEnum(unittest.TestCase):
    def test_affine_transform_2d(self):
        t = np.pi / 3
        affine = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
        image = torch.arange(24).view(1, 1, 4, 6).to(device=torch.device("cpu:0"))
        xform = AffineTransform(
            (3, 4), padding_mode=PaddingMode.BORDER, align_corners=True, mode=InterpolationMode.BILINEAR
        )
        out = xform(image, affine)
        out = out.detach().cpu().numpy()
        expected = [
            [
                [
                    [7.1525574e-07, 4.9999994e-01, 1.0000000e00, 1.4999999e00],
                    [3.8660259e00, 1.3660253e00, 1.8660252e00, 2.3660252e00],
                    [7.7320518e00, 3.0358994e00, 2.7320509e00, 3.2320507e00],
                ]
            ]
        ]
        np.testing.assert_allclose(out, expected, atol=1e-5)

        if torch.cuda.is_available():
            affine = torch.as_tensor(affine, device=torch.device("cuda:0"), dtype=torch.float32)
            image = torch.arange(24).view(1, 1, 4, 6).to(device=torch.device("cuda:0"))
            xform = AffineTransform(
                padding_mode=PaddingMode.BORDER, align_corners=True, mode=InterpolationMode.BILINEAR
            )
            out = xform(image, affine, (3, 4))
            out = out.detach().cpu().numpy()
            expected = [
                [
                    [
                        [7.1525574e-07, 4.9999994e-01, 1.0000000e00, 1.4999999e00],
                        [3.8660259e00, 1.3660253e00, 1.8660252e00, 2.3660252e00],
                        [7.7320518e00, 3.0358994e00, 2.7320509e00, 3.2320507e00],
                    ]
                ]
            ]
            np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_affine_transform_3d(self):
        t = np.pi / 3
        affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
        affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
        image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
        xform = AffineTransform(
            (3, 4, 2), padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
        )
        out = xform(image, affine)
        out = out.detach().cpu().numpy()
        expected = [
            [
                [
                    [[0.00000006, 0.5000001], [2.3660254, 1.3660254], [4.732051, 2.4019241], [5.0, 3.9019237]],
                    [[6.0, 6.5], [8.366026, 7.3660254], [10.732051, 8.401924], [11.0, 9.901924]],
                    [[12.0, 12.5], [14.366026, 13.366025], [16.732052, 14.401924], [17.0, 15.901923]],
                ]
            ],
            [
                [
                    [[24.0, 24.5], [26.366024, 25.366024], [28.732052, 26.401924], [29.0, 27.901924]],
                    [[30.0, 30.5], [32.366028, 31.366026], [34.732048, 32.401924], [35.0, 33.901924]],
                    [[36.0, 36.5], [38.366024, 37.366024], [40.73205, 38.401924], [41.0, 39.901924]],
                ]
            ],
        ]
        np.testing.assert_allclose(out, expected, atol=1e-4)

        if torch.cuda.is_available():
            affine = torch.as_tensor(affine, device=torch.device("cuda:0"), dtype=torch.float32)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cuda:0"))
            xform = AffineTransform(
                padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
            )
            out = xform(image, affine, (3, 4, 2))
            out = out.detach().cpu().numpy()
            expected = [
                [
                    [
                        [[0.00000006, 0.5000001], [2.3660254, 1.3660254], [4.732051, 2.4019241], [5.0, 3.9019237]],
                        [[6.0, 6.5], [8.366026, 7.3660254], [10.732051, 8.401924], [11.0, 9.901924]],
                        [[12.0, 12.5], [14.366026, 13.366025], [16.732052, 14.401924], [17.0, 15.901923]],
                    ]
                ],
                [
                    [
                        [[24.0, 24.5], [26.366024, 25.366024], [28.732052, 26.401924], [29.0, 27.901924]],
                        [[30.0, 30.5], [32.366028, 31.366026], [34.732048, 32.401924], [35.0, 33.901924]],
                        [[36.0, 36.5], [38.366024, 37.366024], [40.73205, 38.401924], [41.0, 39.901924]],
                    ]
                ],
            ]
            np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_ill_affine_transform(self):
        with self.assertRaises(ValueError):  # image too small
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            xform = AffineTransform(
                (3, 4, 2), padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
            )
            xform(torch.as_tensor([1, 2, 3]), affine)

        with self.assertRaises(ValueError):  # output shape too small
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform(
                (3, 4), padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
            )
            xform(image, affine)

        with self.assertRaises(ValueError):  # incorrect affine
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            affine = affine.unsqueeze(0).unsqueeze(0)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform(
                (2, 3, 4), padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
            )
            xform(image, affine)

        with self.assertRaises(ValueError):  # batch doesn't match
            t = np.pi / 3
            affine = [[1, 0, 0, 0], [0.0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]]
            affine = torch.as_tensor(affine, device=torch.device("cpu:0"), dtype=torch.float32)
            affine = affine.unsqueeze(0)
            affine = affine.repeat(3, 1, 1)
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform(
                (2, 3, 4), padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
            )
            xform(image, affine)

        with self.assertRaises(ValueError):  # wrong affine
            affine = torch.as_tensor([[1, 0, 0, 0], [0, 0, 0, 1]])
            image = torch.arange(48).view(2, 1, 4, 2, 3).to(device=torch.device("cpu:0"))
            xform = AffineTransform(
                (2, 3, 4), padding_mode=PaddingMode.BORDER, align_corners=False, mode=InterpolationMode.BILINEAR
            )
            xform(image, affine)


if __name__ == "__main__":
    unittest.main()
