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

from typing import Callable, Optional

import torch
from ignite.metrics import Metric
from ignite.engine import Engine
from monai.inferers.inferer import SimpleInferer
from .workflow import Workflow
from .utils import default_prepare_batch
from .utils import CommonKeys as Keys


class Trainer(Workflow):
    """
    Base class for all kinds of trainers, inherits from Workflow.

    """

    def run(self):
        """
        Execute training based on Ignite Engine.
        If call this function multiple times, it will continuously run from the previous state.

        """
        if self._is_done(self.state):
            self.state.iteration = 0  # to avoid creating new State instance in ignite Engine.run
        super().run()

    def get_train_stats(self):
        return {"total_epochs": self.state.max_epochs, "total_iterations": self.state.epoch_length}


class SupervisedTrainer(Trainer):
    """
    Standard supervised training method with image and label, inherits from trainer and Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        max_epochs (int): the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        train_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        network (Network): to train with this network.
        optimizer (Optimizer): the optimizer associated to the network.
        loss_function (Loss): the loss function associated to the optimizer.
        prepare_batch (Callable): function to parse image and label for current iteration.
        iteration_update (Callable): the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        lr_scheduler (LR Scheduler): the lr scheduler associated to the optimizer.
        inferer (Inferer): inference method that execute model forward on input data, like: SlidingWindow, etc.
        amp (bool): whether to enable auto-mixed-precision training.
        key_train_metric (ignite.metric): compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics (dict): more Ignite metrics that also attach to Ignite Engine.
        train_handlers (list): every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.

    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader,
        network,
        optimizer,
        loss_function,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        lr_scheduler=None,
        inferer=SimpleInferer(),
        amp: bool = True,
        key_train_metric: Optional[Metric] = None,
        additional_metrics=None,
        train_handlers=None,
    ):
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            amp=amp,
            data_loader=train_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
        )

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = inferer

    def _iteration(self, engine: Engine, batchdata):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (dict or array of tensor): input data for this iteration.

        """
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")
        inputs, targets = self.prepare_batch(batchdata)
        inputs, targets = inputs.to(engine.state.device), targets.to(engine.state.device)

        results = dict()
        self.network.train()
        self.optimizer.zero_grad()
        # execute forward computation
        predictions = self.inferer(inputs, self.network)
        # compute loss
        loss = self.loss_function(predictions, targets).mean()
        loss.backward()
        results[Keys.LOSS] = loss.item()
        self.optimizer.step()

        return {Keys.PRED: predictions, Keys.LABEL: targets, Keys.INFO: results}
