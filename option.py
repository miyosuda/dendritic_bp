# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Option(object):
    def __init__(self):
        self.g_lk = 0.1
        self.g_a = 0.8
        self.g_b = 1.0
        self.g_d = 1.0
        self.g_som = 0.8
        self.noise_delta = 0.1

        self.e_exc = 1.0   # Excitatory reversal potential
        self.e_inh = -1.0  # Inhibitory reversal potential

        self.eta_pp_bu = 0.0011875
        self.eta_pp_td = 0.0011875
        self.eta_pi    = 0.0005
        self.eta_ip    = 0.0002375

    @staticmethod
    def get_self_prediction_option(layer_index):
        option = Option()

        option.eta_pp_bu = 0.0011875 # 不使用
        option.eta_pp_td = 0.0011875 # 不使用
        option.eta_pi    = 0.0005
        option.eta_ip    = 0.0002375
        return option

    @staticmethod
    def get_target_prediction_option(layer_index):
        option = Option()
        
        if layer_index == 0:
            option.eta_pp_bu = 0.0011875
            option.eta_pp_td = 0.0011875 # 不使用
            option.eta_pi    = 0.0005
            option.eta_ip    = 0.0002375
        elif layer_index == 1:
            option.eta_pp_bu = 0.0005
            option.eta_pp_td = 0.0011875 # 不使用
            option.eta_pi    = 0.0005
            option.eta_ip    = 0.0011875 # OK
        return option
