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

from monai.losses import GeneralizedDiceLoss

TEST_CASES = [
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.307576,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]]),
            "smooth": 1e-4,
        },
        0.416597,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]),
            "smooth": 0.0,
        },
        0.0,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.469964,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.414507,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True, "reduction": "sum"},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        0.829015,
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True, "reduction": "none"},
        {
            "input": torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]]]),
            "smooth": 1e-4,
        },
        [0.273476, 0.555539],
    ],
    [  # shape: (2, 2, 3), (2, 1, 3)
        {"include_background": False, "to_onehot_y": True},
        {
            "input": torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            "target": torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]),
            "smooth": 1e-8,
        },
        0.0,
    ],
    [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
        {"include_background": True, "do_sigmoid": True},
        {
            "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]),
            "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "smooth": 1e-6,
        },
        0.307576,
    ],
    [  # shape: (1, 2, 4), (1, 1, 4)
        {"include_background": True, "to_onehot_y": True, "do_softmax": True, "w_type": "simple"},
        {
            "input": torch.tensor([[[0.0, 10.0, 10.0, 10.0], [10.0, 0.0, 0.0, 0.0]]]),
            "target": torch.tensor([[[1, 1, 0, 0]]]),
            "smooth": 0.0,
        },
        0.250023,
    ],
]


class TestGeneralizedDiceLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val) -> None:
        result = GeneralizedDiceLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self) -> None:
        loss = GeneralizedDiceLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 2, 3)), torch.ones((4, 5, 6)))

    def test_ill_opts(self) -> None:
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceLoss(do_sigmoid=True, do_softmax=True)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceLoss(reduction="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            GeneralizedDiceLoss(reduction=None)(chn_input, chn_target)

    def test_input_warnings(self) -> None:
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertWarns(Warning):
            loss = GeneralizedDiceLoss(include_background=False)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = GeneralizedDiceLoss(do_softmax=True)
            loss.forward(chn_input, chn_target)
        with self.assertWarns(Warning):
            loss = GeneralizedDiceLoss(to_onehot_y=True)
            loss.forward(chn_input, chn_target)


if __name__ == "__main__":
    unittest.main()
