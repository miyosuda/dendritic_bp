# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
from distutils.util import strtobool

from layer import  LowPassFilter
from network import Network, TargetNetwork

save_dir = "saved"


def train_self_prediction(args, train_iteration=100):
    print("start self prediction task training")
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
    print("start target prediction task training")
    
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

        # Apical部分のエラー電位
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
    print("start non-linear assocication task training")

    def calc_rmse(values, expected_values):
        d = values - expected_values
        mse = np.sum(d * d) / len(values)
        return np.sqrt(mse)
    
    network = Network(force_self_prediction=True)
    target_network = TargetNetwork()
    
    if args.loading:
        network.load(save_dir)
        target_network.load(save_dir)
    
    network.set_nonlinear_association_mode()
    
    dt = 0.1
    lp_filter = LowPassFilter(dt, 3)

    if args.saving:
        network.save(save_dir)
        target_network.save(save_dir)


    save_interval = 500

    for i in range(train_iteration):
        input_values, target_values = target_network.get_training_pair()
        for j in range(1000):
            filtered_input_values = lp_filter.process(input_values)
            network.set_target_firing_rate(target_values)
            network.set_input_firing_rate(filtered_input_values)
            network.update(dt)
            
        #print("error={}".format(np.mean(network.layers[1].v_p_a)))
        #print("target_r={}".format(target_values))
        #print("output_r={}".format(network.layers[2].get_p_activation()))
        #print("target_u={}".format(network.layers[2].u_target))
        #print("output_u={}".format(network.layers[2].u_p))

        rmse = calc_rmse(network.layers[2].get_p_activation(), target_values)

        if args.verbose:
            print("{0}: rmse={1:.4f}".format(i, rmse))

        if args.saving and ((i % save_interval) == (save_interval-1)):
            network.save(save_dir)

    network.clear_target()

    for i in range(100):
        input_values, target_values = target_network.get_training_pair()
        for j in range(1000):
            filtered_input_values = lp_filter.process(input_values)
            network.set_input_firing_rate(filtered_input_values)
            network.update(dt)

        if args.verbose:
            print("target_r={}".format(target_values))
            print("output_r={}".format(network.layers[2].get_p_activation()))

    if args.saving:
        network.save(save_dir)
        

def main(args):
    np.random.seed(seed=args.seed)

    if args.train_type == "self":
        train_self_prediction(args)
    elif args.train_type == "target":
        train_target_prediction(args)
    else:
        train_nonlinear_association(args, train_iteration=args.iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--loading", type=strtobool, default="false")
    parser.add_argument("--saving", type=strtobool, default="true")
    parser.add_argument("--iteration", type=int, default=1000000) # 1000000
    parser.add_argument("--train_type", type=str, default="assoc")
    parser.add_argument("--verbose", type=strtobool, default="false")
    
    # 1 itで、0.1秒, 30hで1080000 it
    args = parser.parse_args()

    main(args)
