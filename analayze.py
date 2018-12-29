# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from model import Layer, LAYER_TYPE_BOTTOM, LAYER_TYPE_HIDDEN, LAYER_TYPE_TOP, LowPassFilter
from option import Option

# TopとBottomだけを接続して分析するためのコード
class Network(object):
    def __init__(self):
        option = Option()

        self.layers = [None] * 2
        self.layers[0] = Layer(pd_unit_size=1, layer_type=LAYER_TYPE_BOTTOM, option=option)
        self.layers[1] = Layer(pd_unit_size=1, layer_type=LAYER_TYPE_TOP,    option=option)
        
        self.layers[0].connect_to(self.layers[1])
        self.set_target_prediction_mode()

    def set_target_prediction_mode(self):
        # Pyramidalのweightを更新する
        for layer in self.layers:
            # ここを変えている
            layer.train_w_pp_bu = True
            layer.train_w_pp_td = False
            layer.train_w_ip = False
            layer.train_w_pi = False

        for i,layer in enumerate(self.layers):
            option = Option.get_target_prediction_option(i)
            layer.set_option(option)

    def update(self, dt):
        for layer in self.layers:
            layer.update_potential(dt)

        for layer in self.layers:
            layer.update_weight(dt)

    def set_input_firing_rate(self, values):
        self.layers[0].set_input_firing_rate(values)

    def set_target_firing_rate(self, values):
        self.layers[1].set_target_firing_rate(values)

def train_target_prediction(network):
    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)

    target_values = np.array([0.8], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)

    network.set_target_firing_rate(target_values)
    network.set_input_firing_rate(values)

    iteration = 2000

    for i in range(iteration):
        for j in range(1000):
            network.update(dt)
            
        du = network.layers[1].u_target - network.layers[1].u_p

        v_p_b = network.layers[1].v_p_b
        u_p = network.layers[1].u_p
        print("du={}, v_p_b={}, u_p={}".format(du, v_p_b, u_p))
        """
        print("upper_r_p={}, upper_v_p_b_hat={}, upper_r_p_b={}, d_w_pp_bu={}".format(
              network.layers[0].debug_upper_r_p,
              network.layers[0].debug_upper_v_p_b_hat,
              network.layers[0].debug_upper_r_p_b,
              network.layers[0].debug_d_w_pp_bu))
        """
        
        
def main():
    np.random.seed(seed=0)
    
    network = Network()
    train_target_prediction(network)


if __name__ == '__main__':
    main()
