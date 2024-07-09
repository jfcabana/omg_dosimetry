# Test the LUT class

import unittest
import pathlib
import os

import omg_dosimetry
from omg_dosimetry.calibration import LUT


class TestLUT(unittest.TestCase):

    def test_demo(self):

        LUT.run_demo(show=False)
        lut_path = os.path.dirname(omg_dosimetry.__file__) + '/demo_files/calibration/Demo_calib.pkl'
        exist = pathlib.Path(lut_path).is_file()

        self.assertTrue(exist)
        