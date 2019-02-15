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

save_dir = "saved"


class TargetNetwork(object):
    def __init__(self):
        self.w0 = np.random.uniform(-1, 1, size=(20,30))
        self.w1 = np.random.uniform(-1, 1, size=(10,20))

    def softplus(self, x):
        return np.log(1.0 + np.exp(x))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def get_training_pair(self):
        input_values = np.random.rand(30)
        h0 = self.softplus(self.w0.dot(input_values))
        target_values = self.sigmoid(self.w1.dot(h0))
        return input_values, target_values
        

class Network(object):
    def __init__(self, force_self_prediction=False):
        option = Option()

        self.layers = [None] * 3
        self.layers[0] = Layer(pd_unit_size=30, 
                               layer_type=LAYER_TYPE_BOTTOM,
                               option=option,
                               force_self_prediction=force_self_prediction)
        self.layers[1] = Layer(pd_unit_size=20,
                               layer_type=LAYER_TYPE_HIDDEN,
                               option=option,
                               force_self_prediction=force_self_prediction)
        self.layers[2] = Layer(pd_unit_size=10,
                               layer_type=LAYER_TYPE_TOP,
                               option=option,
                               force_self_prediction=force_self_prediction)
        
        self.layers[0].connect_to(self.layers[1])
        self.layers[1].connect_to(self.layers[2])

        self.set_self_prediction_mode()

    def set_self_prediction_mode(self):
        # Pyramidalのweightは更新しない
        for layer in self.layers:
            layer.train_w_pp_bu = False
            layer.train_w_pp_td = False
            layer.train_w_ip = True
            layer.train_w_pi = True

        for i,layer in enumerate(self.layers):
            option = Option.get_self_prediction_option(i)
            layer.set_option(option)

    def set_target_prediction_mode(self):
        # Pyramidalのweightを更新する
        for layer in self.layers:
            layer.train_w_pp_bu = True
            layer.train_w_pp_td = False # TopDownのWeightは固定
            layer.train_w_ip = True
            layer.train_w_pi = True

        for i,layer in enumerate(self.layers):
            option = Option.get_target_prediction_option(i)
            layer.set_option(option)

    def set_nonlinear_association_mode(self):
        for layer in self.layers:
            layer.train_w_pp_bu = True
            layer.train_w_pp_td = False # TopDownのWeightは固定
            layer.train_w_ip = True
            layer.train_w_pi = False # SST -> PyramidalのWeightは固定

        for i,layer in enumerate(self.layers):
            option = Option.get_nonlinear_association_option(i)
            layer.set_option(option)

    def update(self, dt):
        for layer in self.layers:
            layer.update_potential(dt)

        for layer in self.layers:
            layer.update_weight(dt)

    def set_input_firing_rate(self, values):
        self.layers[0].set_input_firing_rate(values)

    def set_target_firing_rate(self, values):
        self.layers[2].set_target_firing_rate(values)

    def clear_target(self):
        self.layers[2].clear_target()

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for i in range(2):
            # Top層はSaveするものが無いので対象外
            file_path = os.path.join(dir_name, "layer{}".format(i))
            self.layers[i].save(file_path)

        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        for i in range(2):
            # Top層はLoadするものが無いので対象外
            file_path = os.path.join(dir_name, "layer{}".format(i))
            self.layers[i].load(file_path)

        print("loaded: {}".format(dir_name))


def train_self_prediction(args, train_iteration=100):
    network = Network()
    
    if args.loading:
        network.load(save_dir)
    
    network.set_self_prediction_mode()
    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)

    for i in range(train_iteration):
        # 100ms = 1000 step間値を固定する.
        values = np.random.rand(30)
        for j in range(1000):
            filtered_values = lp_filter.process(values)
            network.set_input_firing_rate(filtered_values)
            network.update(dt)
        print(np.mean(network.layers[1].v_p_a))

    if args.saving:
        network.save(save_dir)


def train_target_prediction(args):
    network = Network()
    
    if args.loading:
        network.load(save_dir)
    
    network.set_target_prediction_mode()
    
    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)

    # ターゲット値を0.0~1.0の乱数で
    target_values = np.random.rand(10)
    # 入力値を0.0~1.0の乱数で
    input_values = np.random.rand(30)

    train_iteration = 200

    for i in range(train_iteration):
        # 100ms = 1000 step間値を固定する.
        for j in range(1000):
            filtered_input_values = lp_filter.process(input_values)
            network.set_target_firing_rate(target_values)
            network.set_input_firing_rate(filtered_input_values)
            network.update(dt)
            
        print("error={}".format(np.mean(network.layers[1].v_p_a)))
        #print("target_r={}".format(target_values))
        #print("output_r={}".format(network.layers[2].get_p_activation()))

        print("target_u={}".format(network.layers[2].u_target))
        print("output_u={}".format(network.layers[2].u_p))

    network.clear_target()

    for i in range(100):
        for j in range(1000):
            filtered_input_values = lp_filter.process(input_values)
            network.set_input_firing_rate(filtered_input_values)
            network.update(dt)
        print("target_r={}".format(target_values))
        print("output_r={}".format(network.layers[2].get_p_activation()))

    if args.saving:
        network.save(save_dir)


def train_nonlinear_association(args, train_iteration=1000):
    network = Network(force_self_prediction=True)
    target_network = TargetNetwork()
    
    if args.loading:
        network.load(save_dir)
    
    network.set_nonlinear_association_mode()
    
    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)

    for i in range(train_iteration):
        input_values, target_values = target_network.get_training_pair()
        for j in range(1000):
            filtered_input_values = lp_filter.process(input_values)
            network.set_target_firing_rate(target_values)
            network.set_input_firing_rate(filtered_input_values)
            network.update(dt)
            
        print("error={}".format(np.mean(network.layers[1].v_p_a)))
        print("target_r={}".format(target_values))
        print("output_r={}".format(network.layers[2].get_p_activation()))

        #print("target_u={}".format(network.layers[2].u_target))
        #print("output_u={}".format(network.layers[2].u_p))

    network.clear_target()

    for i in range(100):
        input_values, target_values = target_network.get_training_pair()
        for j in range(1000):
            filtered_input_values = lp_filter.process(input_values)
            network.set_input_firing_rate(filtered_input_values)
            network.update(dt)
        print("target_r={}".format(target_values))
        print("output_r={}".format(network.layers[2].get_p_activation()))

    if args.saving:
        network.save(save_dir)


def main(args):
    np.random.seed(seed=0)
    
    #train_self_prediction(args)
    #train_target_prediction(args)
    train_nonlinear_association(args, train_iteration=args.iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading", type=strtobool, default="false")
    parser.add_argument("--saving", type=strtobool, default="false")
    parser.add_argument("--iteration", type=int, default=1000)
    # 1000000
    
    args = parser.parse_args()

    main(args)
