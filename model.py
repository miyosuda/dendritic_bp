# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def activation(x):
    return 1.0 / (1.0 + np.exp(-x))


class Layer(object):
    def __init__(self, pd_unit_size, sst_unit_size):
        self.pd_unit_size = pd_unit_size
        self.sst_unit_size = sst_unit_size

        self.upper_layer = None
        self.lower_layer = None
        
        # Pyramidal neuron potentials
        self.u_p = np.zeros([self.pd_unit_size]) # Soma
        
        # SST inter-neuraon potentials
        self.u_i = np.zeros([self.sst_unit_size]) # Soma

    def connect_to(self, upper_layer):
        self.upper_layer = upper_layer
        upper_layer.lower_layer = self
        
        # K -> K+1
        self.w_pp_bu = np.random.uniform(-1, 1, size=(upper_layer.pd_unit_size,
                                                      self.pd_unit_size))
        # k+1 -> K
        self.w_pp_td = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                      upper_layer.pd_unit_size))
        
        # PD -> SST
        self.w_ip = np.random.uniform(-1, 1, size=(self.sst_unit_size,
                                                   self.pd_unit_size))
        # SST -> PD
        self.w_pi = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                   self.sst_unit_size))

    def process(self):
        # Apical, Basalの電位をFiring Rateから計算
        # Piramidal Apical
        v_p_a = self.lower_layer.w_pp_bu.dot(activation(self.lower_layer.u_p))
        # (pd_unit_size)
        
        # Piramidal Basal
        v_p_b = self.w_pp_td.dot(activation(self.upper_layer.u_p)) + \
                self.w_pi.dot(activation(self.u_i))
        # (pd_unit_size)
        
        # SST Basal
        v_i_b = self.w_ip.dot(activation(self.u_p))
        # (sst_unit_size)
