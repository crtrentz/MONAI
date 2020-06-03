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

from monai.transforms import Affine

TEST_CASES = [
    [
        dict(padding_mode="zeros", as_tensor_output=False, device=None),
        {"img": np.arange(4).reshape((1, 2, 2)), "spatial_size": (4, 4)},
        np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.25, 0.0], [0.0, 0.5, 0.75, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
    ],
    [
        dict(rotate_params=[np.pi / 2], padding_mode="zeros", as_tensor_output=False, device=None),
        {"img": np.arange(4).reshape((1, 2, 2)), "spatial_size": (4, 4)},
        np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.75, 0.25, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
    ],
    [
        dict(padding_mode="zeros", as_tensor_output=False, device=None),
        {"img": np.arange(8).reshape((1, 2, 2, 2)), "spatial_size": (4, 4, 4)},
        np.array(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.125, 0.0], [0.0, 0.25, 0.375, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.625, 0.0], [0.0, 0.75, 0.875, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                ]
            ]
        ),
    ],
    [
        dict(rotate_params=[np.pi / 2], padding_mode="zeros", as_tensor_output=False, device=None),
        {"img": np.arange(8).reshape((1, 2, 2, 2)), "spatial_size": (4, 4, 4)},
        np.array(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0], [0.0, 0.375, 0.125, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.75, 0.5, 0.0], [0.0, 0.875, 0.625, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                ]
            ]
        ),
    ],
]


class TestAffine(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_affine(self, input_param, input_data, expected_val) -> None:
        g = Affine(**input_param)
        result = g(**input_data)
        self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected_val))
        if torch.is_tensor(result):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
