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
            option.eta_ip    = 0.0011875
        return option

    def get_nonlinear_association_option(layer_index):
        option = Option()

        option.noise_delta = 0.3 # ノイズ増やす
        
        if layer_index == 0:
            option.eta_pp_bu = 0.00011875
            option.eta_pp_td = 0.0011875 # 不使用 (w固定)
            option.eta_pi    = 0.0005    # 不使用 (w固定)
            option.eta_ip    = 0.00002375
        elif layer_index == 1:
            option.eta_pp_bu = 0.00001
            option.eta_pp_td = 0.0011875 # 不使用 (w固定)
            option.eta_pi    = 0.0005    # 不使用 (w固定)
            option.eta_ip    = 0.00002375
        return option

    """
    def get_nonlinear_association_option_experimental(layer_index):
        option = Option()

        option.noise_delta = 0.0
        
        if layer_index == 0:
            option.eta_pp_bu = 0.00011875 * 10.0
            option.eta_pp_td = 0.0011875 # 不使用 (w固定)
            option.eta_pi    = 0.0005    # 不使用 (w固定)
            option.eta_ip    = 0.00002375 * 10.0
        elif layer_index == 1:
            option.eta_pp_bu = 0.00001 * 10.0
            option.eta_pp_td = 0.0011875 # 不使用 (w固定)
            option.eta_pi    = 0.0005    # 不使用 (w固定)
            option.eta_ip    = 0.00002375 * 10.0
        return option    
    """
