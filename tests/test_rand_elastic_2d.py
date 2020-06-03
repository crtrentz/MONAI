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

from monai.transforms import Rand2DElastic

TEST_CASES = [
    [
        {"spacing": (0.3, 0.3), "magnitude_range": (1.0, 2.0), "prob": 0.0, "as_tensor_output": False, "device": None},
        {"img": torch.ones((3, 3, 3)), "spatial_size": (2, 2)},
        np.ones((3, 2, 2)),
    ],
    [
        {"spacing": (0.3, 0.3), "magnitude_range": (1.0, 2.0), "prob": 0.9, "as_tensor_output": False, "device": None},
        {"img": torch.ones((3, 3, 3)), "spatial_size": (2, 2), "mode": "bilinear"},
        np.array([[[0.0, 0.0], [0.0, 0.04970419]], [[0.0, 0.0], [0.0, 0.04970419]], [[0.0, 0.0], [0.0, 0.04970419]]]),
    ],
    [
        {
            "spacing": (1.0, 1.0),
            "magnitude_range": (1.0, 1.0),
            "scale_range": [1.2, 2.2],
            "prob": 0.9,
            "padding_mode": "border",
            "as_tensor_output": True,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3))},
        torch.tensor(
            [
                [[1.6605, 1.0083], [6.0000, 6.2224]],
                [[10.6605, 10.0084], [15.0000, 15.2224]],
                [[19.6605, 19.0083], [24.0000, 24.2224]],
            ]
        ),
    ],
    [
        {
            "spacing": (0.3, 0.3),
            "magnitude_range": (0.1, 0.2),
            "translate_range": [-0.01, 0.01],
            "scale_range": [0.01, 0.02],
            "prob": 0.9,
            "as_tensor_output": False,
            "device": None,
            "spatial_size": (2, 2),
        },
        {"img": torch.arange(27).reshape((3, 3, 3))},
        np.array(
            [
                [[0.2001334, 1.2563337], [5.2274017, 7.90148]],
                [[8.675412, 6.9098353], [13.019891, 16.850012]],
                [[17.15069, 12.563337], [20.81238, 25.798544]],
            ]
        ),
    ],
]


class TestRand2DElastic(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_2d_elastic(self, input_param, input_data, expected_val) -> None:
        g = Rand2DElastic(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected_val))
        if torch.is_tensor(result):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
