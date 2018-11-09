# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from model import Layer


class LayerTest(unittest.TestCase):
    def test_init(self):
        layer0 = Layer(pd_unit_size=64, sst_unit_size=32)
        layer1 = Layer(pd_unit_size=64, sst_unit_size=32)
        layer2 = Layer(pd_unit_size=64, sst_unit_size=32)
        
        layer0.connect_to(layer1)
        layer1.connect_to(layer2)
        
        layer1.process()
        
        
if __name__ == '__main__':
    unittest.main()
    
