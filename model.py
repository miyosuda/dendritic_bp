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
noise_delta = 1.0

e_exc = 1.0   # Excitatory reversal potential
e_inh = -1.0  # Inhibitory reversal potential

g_som = 1.0

# TODO: etaは層によって違う模様
eta_pp_bu = 1.0
eta_pp_td = 1.0
eta_pi = 1.0
eta_ip = 1.0


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
        g_p_exc = 1.0 # TODO: これは教師信号により変わる変数
        g_p_inh = 1.0 # TODO: これは教師信号により変わる変数
        i_p = g_p_exc * (e_exc - self.u_p) + g_p_inh * (e_inh - self.u_p)
        return i_p

    def calc_i_i(self):
        # トップダウンによるTeaching currentの算出
        g_i_exc =  g_som * (self.upper_layer.u_p - e_inh) / (e_exc - e_inh)
        g_i_inh = -g_som * (self.upper_layer.u_p - e_exc) / (e_exc - e_inh)
        
        # このcurrentにより、SST interneuronは、上位のinter neuronの電位に近づいていく
        i_i = g_i_exc * (e_exc - self.u_i) + g_i_inh * (e_inh - self.u_i)
        return i_i

    def update_potential(self, dt):
        # Apical, Basalの電位をFiring Rateから計算

        # TODO: v_p_a, v_p_b, v_i_b はテンポラリ変数にできそうだが、
        # update_weight()の中でも利用しているので、メンバ変数にしている. 要整理.        
        
        # Piramidal Apical
        # (TopのLayerには無いので0に)
        if self.upper_layer is not None:
            self.v_p_a = self.w_pp_td.dot(activation(self.upper_layer.u_p)) + \
                         self.w_pi.dot(activation(self.u_i))
            # (pd_unit_size)
        else:
            # TODO: 整理して最適化が必要
            self.v_p_a = np.zeros([self.pd_unit_size])
        
        # Piramidal Basal
        if self.lower_layer is not None:
            self.v_p_b = self.lower_layer.w_pp_bu.dot(activation(self.lower_layer.u_p))
        else:
            # TODO: 整理して最適化が必要
            self.v_p_b = np.zeros([self.pd_unit_size])
        
        # SST Basal
        if self.upper_layer is not None:
            # 最上位レイヤーでは、SSTが無い
            self.v_i_b = self.w_ip.dot(activation(self.u_p))
            # (sst_unit_size)
        else:
            # TODO: 整理して最適化が必要
            self.v_i_b = np.zeros([self.sst_unit_size])
        
        # upperからSSTにながれるcurrent
        if self.upper_layer is not None:
            i_i = self.calc_i_i()
        else:
            i_i = 0.0
        
        # 教師信号によるcurrent
        i_p = self.calc_i_p()
        
        # TODO: トップでは、g_aがゼロになる
        
        # 錐体細胞の電位の更新式
        d_u_p = -g_lk * self.u_p + \
                g_b * (self.v_p_b - self.u_p) + \
                g_a * (self.v_p_a - self.u_p) + \
                i_p + \
                noise_delta * noise()
        self.u_p += d_u_p * dt
        
        # SSTインターニューロンの電位の更新式
        d_u_i = -g_lk * self.u_i + \
                g_d * (self.v_i_b - self.u_i) + \
                i_i + \
                noise_delta * noise()
        self.u_i += d_u_i * dt

    def calc_d_weight(self, eta, post, pre):
        # 次元を増やす
        post = post.reshape([1,-1])
        pre = pre.reshape([1,-1])
        d_w = eta * np.matmul(post.T, pre)
        return d_w
    
    def update_weight(self, dt):
        if self.upper_layer is None:
            # 最上位レイヤーでは、Weight更新する部分が無い
            return
        
        # この階層のPiramidalの発火率
        r_p = activation(self.u_p)
        
        # Bottom Up結線のweight更新
        upper_r_p = activation(self.upper_layer.u_p)
        upper_v_p_b_hat = self.upper_layer.v_p_b * (g_b / (g_lk + g_b + g_a))
        d_w_pp_bu = self.calc_d_weight(eta_pp_bu, upper_r_p - upper_v_p_b_hat, r_p)
        self.w_pp_bu += d_w_pp_bu * dt
        
        # TopDownのPlasticyを使う場合
        v_p_td_hat = self.w_pp_td.dot(upper_r_p)
        d_w_pp_td = self.calc_d_weight(eta_pp_td, r_p - v_p_td_hat, upper_r_p)
        self.w_pp_td += d_w_pp_td
        
        # P -> I結線のweight更新
        r_i = activation(self.u_i)
        v_i_b_hat = self.v_i_b * (g_d/g_lk + g_d)
        d_w_ip = self.calc_d_weight(eta_ip, r_i - v_i_b_hat, r_p)
        self.w_ip += d_w_ip * dt
        
        # I -> P結線のweight更新
        # (Apicalの電位を0に近づける)
        v_rest = 0.0
        d_w_pi = self.calc_d_weight(eta_pi, v_rest - self.v_p_a, r_i)
        self.w_pi += d_w_pi * dt
