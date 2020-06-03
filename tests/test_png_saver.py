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
import shutil
import unittest
import torch

from monai.data import PNGSaver


class TestPNGSaver(unittest.TestCase):
    def test_saved_content(self) -> None:
        default_dir = os.path.join(".", "tempdir")
        shutil.rmtree(default_dir, ignore_errors=True)

        saver = PNGSaver(output_dir=default_dir, output_postfix="seg", output_ext=".png")

        meta_data = {"filename_or_obj": ["testfile" + str(i) for i in range(8)]}
        saver.save_batch(torch.randint(1, 200, (8, 1, 2, 2)), meta_data)
        for i in range(8):
            filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.png")
            self.assertTrue(os.path.exists(os.path.join(default_dir, filepath)))
        shutil.rmtree(default_dir)

    def test_saved_content_three_channel(self) -> None:
        default_dir = os.path.join(".", "tempdir")
        shutil.rmtree(default_dir, ignore_errors=True)

        saver = PNGSaver(output_dir=default_dir, output_postfix="seg", output_ext=".png")

        meta_data = {"filename_or_obj": ["testfile" + str(i) for i in range(8)]}
        saver.save_batch(torch.randint(1, 200, (8, 3, 2, 2)), meta_data)
        for i in range(8):
            filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.png")
            self.assertTrue(os.path.exists(os.path.join(default_dir, filepath)))
        shutil.rmtree(default_dir)

    def test_saved_content_spatial_size(self) -> None:

        default_dir = os.path.join(".", "tempdir")
        shutil.rmtree(default_dir, ignore_errors=True)

        saver = PNGSaver(output_dir=default_dir, output_postfix="seg", output_ext=".png")

        meta_data = {
            "filename_or_obj": ["testfile" + str(i) for i in range(8)],
            "spatial_shape": [(4, 4) for i in range(8)],
        }
        saver.save_batch(torch.randint(1, 200, (8, 1, 2, 2)), meta_data)
        for i in range(8):
            filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.png")
            self.assertTrue(os.path.exists(os.path.join(default_dir, filepath)))

        shutil.rmtree(default_dir)


if __name__ == "__main__":
    unittest.main()
