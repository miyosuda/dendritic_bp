# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from model import Layer, LAYER_TYPE_BOTTOM, LAYER_TYPE_HIDDEN, LAYER_TYPE_TOP, LowPassFilter


class LayerTest(unittest.TestCase):
    def test_update(self):
        layer0 = Layer(pd_unit_size=30, layer_type=LAYER_TYPE_BOTTOM)
        layer1 = Layer(pd_unit_size=20, layer_type=LAYER_TYPE_HIDDEN)
        layer2 = Layer(pd_unit_size=10, layer_type=LAYER_TYPE_TOP)
        
        layer0.connect_to(layer1)
        layer1.connect_to(layer2)
        
        dt = 0.1

        for i in range(2):
            layer0.update_potential(dt)
            layer1.update_potential(dt)
            layer2.update_potential(dt)
        
            layer0.update_weight(dt)
            layer1.update_weight(dt)
            layer2.update_weight(dt)
        
    def test_calc_d_weight(self):
        layer = Layer(pd_unit_size=10, layer_type=LAYER_TYPE_HIDDEN)
        
        eta = 1.0
        post = np.zeros([10])
        pre = np.zeros([20])
        d_w = layer.calc_d_weight(eta, post, pre)
        self.assertEqual(d_w.shape, (10, 20))

    def test_low_pass_filter(self):
        lowpass_filter = LowPassFilter(0.1, 3.0)
        value = np.ones([10], dtype=np.float32)
        
        value0 = lowpass_filter.process(value)
        self.assertEqual(value0.shape, (10,))

        value1 = lowpass_filter.process(value)
        self.assertEqual(value1.shape, (10,))


if __name__ == '__main__':
    unittest.main()
