# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from model import Layer, LAYER_TYPE_BOTTOM, LAYER_TYPE_HIDDEN, LAYER_TYPE_TOP, LowPassFilter


class Network(object):
    def __init__(self):
        self.layer0 = Layer(pd_unit_size=30, layer_type=LAYER_TYPE_BOTTOM)
        self.layer1 = Layer(pd_unit_size=20, layer_type=LAYER_TYPE_HIDDEN)
        self.layer2 = Layer(pd_unit_size=10, layer_type=LAYER_TYPE_TOP)
        
        self.layer0.connect_to(self.layer1)
        self.layer1.connect_to(self.layer2)

    def update(self, dt):
        self.layer0.update_potential(dt)
        self.layer1.update_potential(dt)
        self.layer2.update_potential(dt)
        
        self.layer0.update_weight(dt)
        self.layer1.update_weight(dt)
        self.layer2.update_weight(dt)

    def set_sensor_input(self, values):
        self.layer0.set_sensor_input(values)


def main():
    network = Network()
    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)
    
    for i in range(100):
        # 100ms = 1000 step間値を固定する.
        values = np.random.rand(30)
        for j in range(1000):
            print(values)
            
            values = lp_filter.process(values)
            network.set_sensor_input(values)
            network.update(dt)
            
            
if __name__ == '__main__':
    main()
