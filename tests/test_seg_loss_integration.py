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
import torch.nn as nn
import torch.optim as optim
from parameterized import parameterized

from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss, TverskyLoss

TEST_CASES = [
    [DiceLoss, {"to_onehot_y": True, "squared_pred": True}, {"smooth": 1e-4}],
    [DiceLoss, {"to_onehot_y": True, "do_sigmoid": True}, {}],
    [DiceLoss, {"to_onehot_y": True, "do_softmax": True}, {}],
    [FocalLoss, {"gamma": 1.5, "weight": torch.tensor([1, 2])}, {}],
    [FocalLoss, {"gamma": 1.5}, {}],
    [GeneralizedDiceLoss, {"to_onehot_y": True, "do_softmax": True}, {}],
    [GeneralizedDiceLoss, {"to_onehot_y": True, "do_sigmoid": True}, {}],
    [GeneralizedDiceLoss, {"to_onehot_y": True, "do_sigmoid": True, "w_type": "simple"}, {}],
    [GeneralizedDiceLoss, {"to_onehot_y": True, "do_sigmoid": True, "w_type": "uniform"}, {}],
    [TverskyLoss, {"to_onehot_y": True, "do_softmax": True, "alpha": 0.8, "beta": 0.2}, {}],
    [TverskyLoss, {"to_onehot_y": True, "do_softmax": True, "alpha": 1.0, "beta": 0.0}, {}],
]


class TestSegLossIntegration(unittest.TestCase):
    def setUp(self) -> None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self) -> None:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @parameterized.expand(TEST_CASES)
    def test_convergence(self, loss_type, loss_args, forward_args) -> None:
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.001
        max_iter = 40

        # define a simple 3d example
        target_seg = torch.tensor(
            [
                [
                    # raw 0
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    # raw 1
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    # raw 2
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                ]
            ],
            device=self.device,
        )
        target_seg = torch.unsqueeze(target_seg, dim=0)
        image = 12 * target_seg + 27
        image = image.float().to(self.device)
        num_classes = 2
        num_voxels = 3 * 4 * 4

        # define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer_1 = nn.Linear(num_voxels, 200)
                self.acti = nn.ReLU()
                self.layer_2 = nn.Linear(200, num_voxels * num_classes)

            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer_1(x)
                x = self.acti(x)
                x = self.layer_2(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        # initialise the network
        net = OnelayerNet().to(self.device)

        # initialize the loss
        loss = loss_type(**loss_args)

        # initialize a SGD optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        loss_history = []

        first_iter: bool = True
        # train the network
        for iter_i in range(max_iter):
            # set the gradient to zero
            optimizer.zero_grad()

            # forward pass
            output = net(image)
            if first_iter:
                first_iter = False
                init_output = torch.argmax(output, 1).detach().cpu().numpy()
            loss_val = loss(output, target_seg, **forward_args)

            if iter_i % 10 == 0:
                pred = torch.argmax(output, 1).detach().cpu().numpy()
                gt = target_seg.detach().cpu().numpy()[:, 0]
                print(f"{loss_type.__name__} iter: {iter_i}, acc: {np.sum(pred == gt) / np.prod(pred.shape)}")

            # backward pass
            loss_val.backward()
            optimizer.step()

            # stats
            loss_history.append(loss_val.item())

        pred = torch.argmax(output, 1).detach().cpu().numpy()
        target = target_seg.detach().cpu().numpy()[:, 0]
        # initial predictions are bad
        self.assertTrue(not np.allclose(init_output, target))
        # final predictions are good
        np.testing.assert_allclose(pred, target)


if __name__ == "__main__":
    unittest.main()
