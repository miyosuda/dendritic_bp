# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def activation(x):
    return 1.0 / (1.0 + np.exp(-x))

def noise(size):
    return np.random.normal(size=size)


class LowPassFilter(object):
    def __init__(self, dt, time_constant):
        self.last_value = None
        # alpha = dt / (RC + dt)
        self.alpha = dt / (time_constant + dt)

    def process(self, value):
        if self.last_value is None:
            self.last_value = value * self.alpha
        else:
            self.last_value = self.last_value * (1.0-self.alpha) + value * self.alpha
        return self.last_value

# 定数
g_lk = 0.1
g_a = 0.8
g_b = 1.0
g_d = 1.0
g_som = 0.8
noise_delta = 0.1

e_exc = 1.0   # Excitatory reversal potential
e_inh = -1.0  # Inhibitory reversal potential

# TODO: etaは層によって変える必要あり
eta_pp_bu = 0.0011875
eta_pp_td = 0.0011875
eta_pi = 0.0002375
eta_ip = 0.0005

LAYER_TYPE_BOTTOM = 0
LAYER_TYPE_HIDDEN = 1
LAYER_TYPE_TOP    = 2


class Layer(object):
    def __init__(self, pd_unit_size, layer_type):
        self.pd_unit_size = pd_unit_size # 錐体細胞のユニット数

        self.layer_type = layer_type
        
        self.upper_layer = None
        self.lower_layer = None

        if self.layer_type is not LAYER_TYPE_BOTTOM:
            # Pyramidal neuron Soma potentials
            self.u_p = np.zeros([self.pd_unit_size])
        else:
            # 最下層は入力としての発火率を入れる
            # TODO: ここは発火率ではなくて、電位で指定かもしれない.
            self.external_r = np.zeros([self.pd_unit_size])

    def set_sensor_input(self, values):
        self.external_r = values

    def connect_to(self, upper_layer):
        self.upper_layer = upper_layer
        upper_layer.lower_layer = self

        if self.layer_type is not LAYER_TYPE_BOTTOM:
            # 最下層レイヤーにはSST Interneuronが無い
            self.sst_unit_size = upper_layer.pd_unit_size # 錐体細胞のユニット数
            # SST inter-neuraon Soma potentials
            self.u_i = np.zeros([self.sst_unit_size])

            # PD -> SST
            self.w_ip = np.random.uniform(-1, 1, size=(self.sst_unit_size,
                                                       self.pd_unit_size))
            # SST -> PD
            self.w_pi = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                       self.sst_unit_size))

            # Low pass filter
            self.filter_d_w_ip = LowPassFilter(0.1, 30)
            self.filter_d_w_pi = LowPassFilter(0.1, 30)
            
        # K -> K+1
        self.w_pp_bu = np.random.uniform(-1, 1, size=(upper_layer.pd_unit_size,
                                                      self.pd_unit_size))
        # K+1 -> K
        self.w_pp_td = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                      upper_layer.pd_unit_size))

    def get_p_activation(self):
        if self.layer_type == LAYER_TYPE_BOTTOM:
            # TODO: ここは発火率ではなくて、電位で指定かもしれない.
            return self.external_r
        else:
            return activation(self.u_p)
        
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
        if self.layer_type == LAYER_TYPE_HIDDEN:
            self.v_p_a = self.w_pp_td.dot(activation(self.upper_layer.u_p)) + \
                         self.w_pi.dot(activation(self.u_i))
            # (pd_unit_size)
        
        # Piramidal Basal
        if self.layer_type == LAYER_TYPE_HIDDEN or self.layer_type == LAYER_TYPE_TOP:
            self.v_p_b = self.lower_layer.w_pp_bu.dot(self.lower_layer.get_p_activation())
        
        # SST Basal
        if self.layer_type == LAYER_TYPE_HIDDEN:
            self.v_i_b = self.w_ip.dot(activation(self.u_p))
            # (sst_unit_size)

        # 錐体細胞の電位の更新式
        if self.layer_type == LAYER_TYPE_TOP:
            # 教師信号によるcurrent
            # TODO:
            #i_p = self.calc_i_p()
            i_p = 0.0
            
            # 最終層にはApical無いがi_pがある
            d_u_p = -g_lk * self.u_p + \
                    g_b * (self.v_p_b - self.u_p) + \
                    i_p + \
                    noise_delta * noise(len(self.u_p))
            self.u_p += d_u_p * dt
        elif self.layer_type == LAYER_TYPE_HIDDEN:
            # 中間層にはApicalがあるがi_pが無い
            d_u_p = -g_lk * self.u_p + \
                    g_b * (self.v_p_b - self.u_p) + \
                    g_a * (self.v_p_a - self.u_p) + \
                    noise_delta * noise(len(self.u_p))
            self.u_p += d_u_p * dt

        # SSTインターニューロンの電位の更新式
        if self.layer_type == LAYER_TYPE_HIDDEN:
            # upperからSSTにながれるcurrent
            i_i = self.calc_i_i()
            d_u_i = -g_lk * self.u_i + \
                    g_d * (self.v_i_b - self.u_i) + \
                    i_i + \
                    noise_delta * noise(len(self.u_i))
            self.u_i += d_u_i * dt

    def calc_d_weight(self, eta, post, pre):
        # 次元を増やす
        post = post.reshape([1,-1])
        pre = pre.reshape([1,-1])
        d_w = eta * np.matmul(post.T, pre)
        return d_w
    
    def update_weight(self, dt):
        if self.layer_type == LAYER_TYPE_TOP:
            # 最上位レイヤーでは、Weight更新する部分が無い
            return
        
        r_p = self.get_p_activation()
        
        if self.layer_type == LAYER_TYPE_HIDDEN or self.layer_type == LAYER_TYPE_BOTTOM:
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
        if self.layer_type == LAYER_TYPE_HIDDEN:
            r_i = activation(self.u_i)
            v_i_b_hat = self.v_i_b * (g_d/g_lk + g_d)
            d_w_ip = self.calc_d_weight(eta_ip, r_i - v_i_b_hat, r_p)

            # Low pass filter適用
            d_w_ip = self.filter_d_w_ip.process(d_w_ip)
            
            self.w_ip += d_w_ip * dt
        
            # I -> P結線のweight更新
            # (Apicalの電位を0に近づける)
            v_rest = 0.0
            d_w_pi = self.calc_d_weight(eta_pi, v_rest - self.v_p_a, r_i)

            # Low pass filter適用
            d_w_pi = self.filter_d_w_pi.process(d_w_pi)
            
            self.w_pi += d_w_pi * dt
