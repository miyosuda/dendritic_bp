# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import argparse
from distutils.util import strtobool

from model import Layer, LAYER_TYPE_BOTTOM, LAYER_TYPE_HIDDEN, LAYER_TYPE_TOP, LowPassFilter
from option import Option


class Network(object):
    def __init__(self):
        option = Option()

        self.layers = [None] * 3
        self.layers[0] = Layer(pd_unit_size=30, layer_type=LAYER_TYPE_BOTTOM, option=option)
        self.layers[1] = Layer(pd_unit_size=20, layer_type=LAYER_TYPE_HIDDEN, option=option)
        self.layers[2] = Layer(pd_unit_size=10, layer_type=LAYER_TYPE_TOP,    option=option)
        
        self.layers[0].connect_to(self.layers[1])
        self.layers[1].connect_to(self.layers[2])

        # Pyramidalのweightは更新しない
        for layer in self.layers:
            layer.train_w_pp_bu = False
            layer.train_w_pp_td = False

    def update(self, dt):
        for layer in self.layers:        
            layer.update_potential(dt)

        for layer in self.layers:
            layer.update_weight(dt)

    def set_sensor_input(self, values):
        self.layers[0].set_sensor_input(values)

    # TODO: 仮処理
    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path0 = os.path.join(dir_name, "layer0")
        self.layers[0].save(file_path0)
        file_path1 = os.path.join(dir_name, "layer1")
        self.layers[1].save(file_path1)

    def load(self, dir_name):
        file_path0 = os.path.join(dir_name, "layer0")
        self.layers[0].load(file_path0)
        file_path1 = os.path.join(dir_name, "layer1")
        self.layers[1].load(file_path1)


def main(args):
    np.random.seed(seed=0)
    save_dir = "saved"
    
    network = Network()
    if args.loading:
        network.load(save_dir)

    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)
    
    for i in range(100):
        # 100ms = 1000 step間値を固定する.
        values = np.random.rand(30)
        for j in range(1000):
            filtered_values = lp_filter.process(values)
            network.set_sensor_input(filtered_values)
            network.update(dt)

        print(np.mean(network.layers[1].v_p_a))

    if args.saving:
        network.save(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading", type=strtobool, default="true")
    parser.add_argument("--saving", type=strtobool, default="true")
    
    args = parser.parse_args()

    main(args)
