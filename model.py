# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def activation(x):
    return 1.0 / (1.0 + np.exp(-x))

def noise():
    return 0.0

# 定数
g_lk = 1.0
g_b = 1.0
g_a = 1.0
g_d = 1.0
delta = 1.0

e_exc = 1.0   # Excitatory reversal potential
e_inh = -1.0  # Inhibitory reversal potential

g_som = 1.0


class Layer(object):
    def __init__(self, pd_unit_size, sst_unit_size):
        self.pd_unit_size = pd_unit_size # 錐体細胞のユニット数
        self.sst_unit_size = sst_unit_size # 錐体細胞のユニット数
        
        self.upper_layer = None
        self.lower_layer = None
        
        # Pyramidal neuron potentials
        # 錐体細胞の電位
        self.u_p = np.zeros([self.pd_unit_size]) # Soma
        
        # SST inter-neuraon potentials
        # SSTインターニューロンの電位
        self.u_i = np.zeros([self.sst_unit_size]) # Soma
        
    def connect_to(self, upper_layer):
        self.upper_layer = upper_layer
        upper_layer.lower_layer = self
        
        # K -> K+1
        self.w_pp_bu = np.random.uniform(-1, 1, size=(upper_layer.pd_unit_size,
                                                      self.pd_unit_size))
        # K+1 -> K
        self.w_pp_td = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                      upper_layer.pd_unit_size))
        
        # PD -> SST
        self.w_ip = np.random.uniform(-1, 1, size=(self.sst_unit_size,
                                                   self.pd_unit_size))
        # SST -> PD
        self.w_pi = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                   self.sst_unit_size))
        
    def calc_i_p(self):
        # トップLayerにおける教師信号によるcurrentを算出
        g_p_exc = 1.0 # これは教師信号により変わる変数？
        g_p_inh = 1.0 # これは教師信号により変わる変数？
        i_p = g_p_exc * (e_exc - self.u_p) + g_p_inh * (e_inh - self.u_p)
        return i_p

    def calc_i_i(self):
        # トップダウンによるTeaching currentの算出
        g_i_exc =  g_som * (self.upper_layer.u_p - e_inh) / (e_exc - e_inh)
        g_i_inh = -g_som * (self.upper_layer.u_p - e_exc) / (e_exc - e_inh)
        
        # このcurrentにより、SST interneuronは、上位のinter neuronの電位に近づいていく
        # TODO: Check
        i_i = g_i_exc * (e_exc - self.u_i) + g_i_inh * (e_inh - self.u_i)
        return i_i

    def update_potential(self, dt):
        # Apical, Basalの電位をFiring Rateから計算
        
        # Piramidal Apical
        # (TopのLayerには無いので0に)
        v_p_a = self.w_pp_td.dot(activation(self.upper_layer.u_p)) + \
                self.w_pi.dot(activation(self.u_i))
        # (pd_unit_size)
        
        # Piramidal Basal
        v_p_b = self.lower_layer.w_pp_bu.dot(activation(self.lower_layer.u_p))
        # (pd_unit_size)
        
        # SST Basal
        v_i_b = self.w_ip.dot(activation(self.u_p))
        # (sst_unit_size)
        
        # upperからSSTにながれるcurrent
        i_i = self.calc_i_i()
        
        # 教師信号によるcurrent
        i_p = self.calc_i_p()
        
        # TODO: トップでは、g_aがゼロになる
        
        # 錐体細胞の電位の更新式
        d_u_p = -g_lk * self.u_p + \
                g_b * (v_p_b - self.u_p) + \
                g_a * (v_p_a - self.u_p) + \
                i_p + \
                delta * noise()
        self.u_p += d_u_p * dt
        
        # SSTインターニューロンの電位の更新式
        d_u_i = -g_lk * self.u_i + \
                g_d * (v_i_b - self.u_i) + \
                i_i + \
                delta * noise()

    def update_weight(self, dt):
        pass
        
    def update(self, dt):
        self.update_potential(dt)
        self.update_weight(dt)
