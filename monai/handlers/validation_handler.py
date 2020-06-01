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

from ignite.engine import Events, Engine
from monai.engines import Evaluator


class ValidationHandler:
    """
    Attach validator to the trainer engine in Ignite.
    It can support to execute validation every N epochs or every N iterations.

    """

    def __init__(self, validator: Evaluator, interval: int, epoch_level: bool = True):
        """
        Args:
            validator (Evaluator): run the validator when trigger validation, suppose to be Evaluator.
            interval (int): do validation every N epochs or every N iterations during training.
            epoch_level (bool): execute validation every N epochs or N iterations.
                `True` is epoch level, `False` is iteration level.

        """
        if not isinstance(validator, Evaluator):
            raise ValueError("validator must be Evaluator ignite engine.")
        self.validator = validator
        self.interval = interval
        self.epoch_level = epoch_level

    def attach(self, engine: Engine):
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine):
        self.validator.run(engine.state.epoch)
