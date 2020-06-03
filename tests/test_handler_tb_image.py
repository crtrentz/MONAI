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

import glob
import os
import shutil
import unittest
import tempfile
import numpy as np
import torch
from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.handlers import TensorBoardImageHandler

TEST_CASES = [
    [[20, 20]],
    [[2, 20, 20]],
    [[3, 20, 20]],
    [[20, 20, 20]],
    [[2, 20, 20, 20]],
    [[2, 2, 20, 20, 20]],
]


class TestHandlerTBImage(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_tb_image_shape(self, shape) -> None:
        tempdir = tempfile.mkdtemp()
        shutil.rmtree(tempdir, ignore_errors=True)

        # set up engine
        def _train_func(engine, batch):
            return torch.zeros((1, 1, 10, 10))

        engine = Engine(_train_func)

        # set up testing handler
        stats_handler = TensorBoardImageHandler(log_dir=tempdir)
        engine.add_event_handler(Events.ITERATION_COMPLETED, stats_handler)

        data = zip(np.random.normal(size=(10, 4, *shape)), np.random.normal(size=(10, 4, *shape)))
        engine.run(data, epoch_length=10, max_epochs=1)

        self.assertTrue(os.path.exists(tempdir))
        self.assertTrue(len(glob.glob(tempdir)) > 0)
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
