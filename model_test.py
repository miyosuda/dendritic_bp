# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from model import Layer


class LayerTest(unittest.TestCase):
    def test_init(self):
        # sst_unit_sizeは上の階層のpd_unitサイズに等しくなる必要あり.
        layer0 = Layer(pd_unit_size=64, sst_unit_size=32)
        layer1 = Layer(pd_unit_size=32, sst_unit_size=10)
        layer2 = Layer(pd_unit_size=10, sst_unit_size=5)
        
        layer0.connect_to(layer1)
        layer1.connect_to(layer2)

        dt = 0.01
        layer1.update(dt)
        
        
if __name__ == '__main__':
    unittest.main()
    
