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

        layer0.update_potential(dt)
        layer1.update_potential(dt)
        layer2.update_potential(dt)
        
        layer0.update_weight(dt)
        layer1.update_weight(dt)
        layer2.update_weight(dt)

    def test_calc_d_weight(self):
        layer = Layer(pd_unit_size=10, sst_unit_size=10)
        
        eta = 1.0
        post = np.zeros([10])
        pre = np.zeros([20])
        d_w = layer.calc_d_weight(eta, post, pre)
        self.assertEqual(d_w.shape, (10, 20))
        
        
if __name__ == '__main__':
    unittest.main()
