# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from network import TargetNetwork, Network


class TargetNetworkTest(unittest.TestCase):
    def test_init(self):
        np.random.seed(seed=0)
        target_network = TargetNetwork()

        for i in range(10):
            input_values, target_values = target_network.get_training_pair()
            self.assertEqual(input_values.shape, (30, ))
            self.assertEqual(target_values.shape, (10, ))

            print(target_values)

    """
    def test_save(self):
        np.random.seed(seed=0)
        target_network = TargetNetwork()

        target_network.save("tmp")
        target_network.load("tmp")
    """


if __name__ == '__main__':
    unittest.main()
