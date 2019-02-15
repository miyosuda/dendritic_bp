# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os


def activation(x):
    """ Sigmoid activation function. """
    return 1.0 / (1.0 + np.exp(-x))

def inv_activation(y):
    """ Inverse of sigmoid activation function. """
    return np.log(y) - np.log(1-y)

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

LAYER_TYPE_BOTTOM = 0
LAYER_TYPE_HIDDEN = 1
LAYER_TYPE_TOP    = 2

class Layer(object):
    def __init__(self, pd_unit_size, layer_type, option, force_self_prediction=False):
        self.pd_unit_size = pd_unit_size # 錐体細胞のユニット数

        self.layer_type = layer_type
        
        self.upper_layer = None
        self.lower_layer = None

        self.set_option(option)

        # 強制的にSelf Prediction状態にweightを初期かするかどうか
        self.force_self_prediction = force_self_prediction

        # 各Weightを更新するかどうかフラグ
        self.train_w_pp_bu = True
        self.train_w_pp_td = True
        self.train_w_ip = True
        self.train_w_pi = True

        if self.layer_type is not LAYER_TYPE_BOTTOM:
            # Pyramidal neuron Soma potentials
            self.u_p = np.zeros([self.pd_unit_size])
        else:
            # 最下層は入力としての発火率を入れる
            self.external_r = np.zeros([self.pd_unit_size])

        if self.layer_type is LAYER_TYPE_TOP:
            self.clear_target()

    def set_option(self, option):
        self.option = option

    def set_input_firing_rate(self, values):
        """ Input値の発火率を指定 """
        self.external_r = values

    def set_target_potential(self, u_target):
        """ ターゲットの電位を指定 """
        self.u_target = u_target

    def set_target_firing_rate(self, r_target):
        """ ターゲットの発火率を指定 """
        # ターゲットの発火率からターゲットの電位を逆算して求める
        self.u_target = inv_activation(r_target)

    def clear_target(self):
        """ ターゲット指定をクリアする. """
        self.u_target = None

    def connect_to(self, upper_layer):
        self.upper_layer = upper_layer
        upper_layer.lower_layer = self

        # K -> K+1
        self.w_pp_bu = np.random.uniform(-1, 1, size=(upper_layer.pd_unit_size,
                                                      self.pd_unit_size))
        # K+1 -> K
        self.w_pp_td = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                      upper_layer.pd_unit_size))

        if self.force_self_prediction:
            # scalingする
            self.w_pp_bu = self.w_pp_bu * 0.1

        if self.layer_type is not LAYER_TYPE_BOTTOM:
            # 最下層レイヤーにはSST Interneuronが無い
            self.sst_unit_size = upper_layer.pd_unit_size # 錐体細胞のユニット数
            # SST inter-neuraon Soma potentials
            self.u_i = np.zeros([self.sst_unit_size])

            if self.force_self_prediction:
                # 強制的にSelf Prediction Stateにする場合
                # PD -> SST
                self.w_ip = self.w_pp_bu.copy()
                # SST -> PD
                self.w_pi = -self.w_pp_td
            else:
                # PD -> SST
                self.w_ip = np.random.uniform(-1, 1, size=(self.sst_unit_size,
                                                           self.pd_unit_size))
                # SST -> PD
                self.w_pi = np.random.uniform(-1, 1, size=(self.pd_unit_size,
                                                           self.sst_unit_size))

            # Low pass filter
            self.filter_d_w_ip = LowPassFilter(0.1, 30)
            self.filter_d_w_pi = LowPassFilter(0.1, 30)

        if self.layer_type is not LAYER_TYPE_TOP:
            if self.force_self_prediction:
                # Non-linear associiation modeではbottom up weightにもlow pass入れる
                self.filter_d_w_pp_bu = LowPassFilter(0.1, 30)
            else:
                self.filter_d_w_pp_bu = None

    def get_p_activation(self):
        # Pyramidal CellのSomaの発火率を得る.
        if self.layer_type == LAYER_TYPE_BOTTOM:
            # 外部からセンサ入力を指定している場合
            return self.external_r
        else:
            # 通常の場合
            return activation(self.u_p)
        
    def calc_i_p(self):
        if self.u_target is None:
            # 外部からの入力が無い場合はカレントは0を返す.
            return 0.0
        
        # トップLayerにおける教師信号によるcurrentを算出
        g_p_exc =  self.option.g_som * \
                   (self.u_target - self.option.e_inh) / \
                   (self.option.e_exc - self.option.e_inh)
        g_p_inh = -self.option.g_som * \
                  (self.u_target - self.option.e_exc) / \
                  (self.option.e_exc - self.option.e_inh)

        # このcurrentにより、u_pは、u_targetに近づいていく
        i_p = g_p_exc * (self.option.e_exc - self.u_p) + \
              g_p_inh * (self.option.e_inh - self.u_p)
        return i_p

    def calc_i_i(self):
        # トップダウンによるTeaching currentの算出
        g_i_exc =  self.option.g_som * \
                   (self.upper_layer.u_p - self.option.e_inh) / \
                   (self.option.e_exc - self.option.e_inh)
        g_i_inh = -self.option.g_som * \
                  (self.upper_layer.u_p - self.option.e_exc) / \
                  (self.option.e_exc - self.option.e_inh)
        
        # このcurrentにより、SST interneuronは、上位のinter neuronの電位に近づいていく
        i_i = g_i_exc * (self.option.e_exc - self.u_i) + \
              g_i_inh * (self.option.e_inh - self.u_i)
        return i_i

    def update_potential(self, dt):
        # Apical, Basalの電位をFiring Rateから計算

        # TODO: v_p_a, v_p_b, v_i_b はテンポラリ変数にできそうだが、
        # update_weight()の中でも利用しているので、メンバ変数にしている. 要整理.
        
        # Pyramidal Apical
        if self.layer_type == LAYER_TYPE_HIDDEN:
            self.v_p_a = self.w_pp_td.dot(activation(self.upper_layer.u_p)) + \
                         self.w_pi.dot(activation(self.u_i))
            # (pd_unit_size)

        # Bottomでは、v_p_aの更新をしていない.

        # Pyramidal Basal
        if self.layer_type == LAYER_TYPE_HIDDEN or self.layer_type == LAYER_TYPE_TOP:
            self.v_p_b = self.lower_layer.w_pp_bu.dot(self.lower_layer.get_p_activation())
        
        # SST Basal
        if self.layer_type == LAYER_TYPE_HIDDEN:
            self.v_i_b = self.w_ip.dot(activation(self.u_p))
            # (sst_unit_size)

        # 錐体細胞の電位の更新式
        if self.layer_type == LAYER_TYPE_TOP:
            # 教師信号によるcurrent
            i_p = self.calc_i_p()
            
            # 最終層にはApical無いがi_pがある
            d_u_p = -self.option.g_lk * self.u_p + \
                    self.option.g_b * (self.v_p_b - self.u_p) + \
                    i_p + \
                    self.option.noise_delta * noise(len(self.u_p))
            self.u_p += d_u_p * dt
        elif self.layer_type == LAYER_TYPE_HIDDEN:
            # 中間層にはApicalがあるがi_pが無い
            d_u_p = -self.option.g_lk * self.u_p + \
                    self.option.g_b * (self.v_p_b - self.u_p) + \
                    self.option.g_a * (self.v_p_a - self.u_p) + \
                    self.option.noise_delta * noise(len(self.u_p))
            self.u_p += d_u_p * dt
        
        # Bottomのu_pはここでは更新しない (外部から与えられるので)

        # SSTインターニューロンの電位の更新式
        if self.layer_type == LAYER_TYPE_HIDDEN:
            # upperからSSTにながれるcurrent
            i_i = self.calc_i_i()
            d_u_i = -self.option.g_lk * self.u_i + \
                    self.option.g_d * (self.v_i_b - self.u_i) + \
                    i_i + \
                    self.option.noise_delta * noise(len(self.u_i))
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

        # Self Predicting state学習時は固定で学習しない
        if self.layer_type == LAYER_TYPE_HIDDEN or self.layer_type == LAYER_TYPE_BOTTOM:
            if self.train_w_pp_bu:
                # Bottom Up結線のweight更新
                upper_r_p = activation(self.upper_layer.u_p)

                if self.upper_layer.layer_type == LAYER_TYPE_TOP:
                    # 上の層がTopの場合はg_a=0としてattenulationを計算しないといけない.
                    attenuation_rate = (self.option.g_b / \
                                        (self.option.g_lk + self.option.g_b))
                else:
                    attenuation_rate = (self.option.g_b / \
                                        (self.option.g_lk + self.option.g_b + self.option.g_a))
                upper_v_p_b_hat = self.upper_layer.v_p_b * attenuation_rate
                upper_r_p_b = activation(upper_v_p_b_hat)
                d_w_pp_bu = self.calc_d_weight(self.option.eta_pp_bu,
                                               upper_r_p - upper_r_p_b, r_p)
                
                if self.filter_d_w_pp_bu is not None:
                    # bottom upにもLowPassかける場合
                    d_w_pp_bu = self.filter_d_w_pp_bu.process(d_w_pp_bu)
                self.w_pp_bu += d_w_pp_bu * dt
        
            if self.train_w_pp_td:
                # TopDownのPlasticyを使う場合
                v_p_td_hat = self.w_pp_td.dot(upper_r_p)
                r_p_td = activation(v_p_td_hat)
                d_w_pp_td = self.calc_d_weight(self.option.eta_pp_td,
                                               r_p - r_p_td, upper_r_p)
                                               
                self.w_pp_td += d_w_pp_td
        
        if self.layer_type == LAYER_TYPE_HIDDEN:
            if self.train_w_ip:
                # P -> I結線のweight更新
                r_i = activation(self.u_i)
                v_i_b_hat = self.v_i_b * (self.option.g_d/self.option.g_lk + self.option.g_d)
                r_i_b = activation(v_i_b_hat)
                d_w_ip = self.calc_d_weight(self.option.eta_ip, r_i - r_i_b, r_p)

                # Low pass filter適用
                d_w_ip = self.filter_d_w_ip.process(d_w_ip)
                self.w_ip += d_w_ip * dt

            if self.train_w_pi:
                # I -> P結線のweight更新
                # (Apicalの電位を0に近づける)
                v_rest = 0.0
                d_w_pi = self.calc_d_weight(self.option.eta_pi, v_rest - self.v_p_a, r_i)

                # Low pass filter適用
                d_w_pi = self.filter_d_w_pi.process(d_w_pi)
                self.w_pi += d_w_pi * dt

    def save(self, file_path):
        if self.layer_type == LAYER_TYPE_HIDDEN:
            np.savez_compressed(file_path,
                                w_pp_bu=self.w_pp_bu,
                                w_pp_td=self.w_pp_td,
                                w_ip=self.w_ip,
                                w_pi=self.w_pi)
        elif self.layer_type == LAYER_TYPE_BOTTOM:
            np.savez_compressed(file_path,
                                w_pp_bu=self.w_pp_bu,
                                w_pp_td=self.w_pp_td)

    def load(self, file_path):
        real_file_path = "{}.npz".format(file_path)
        if not os.path.exists(real_file_path):
            print("saved file not found")
            return
        
        data = np.load(real_file_path)
        if self.layer_type == LAYER_TYPE_HIDDEN:
            self.w_pp_bu = data["w_pp_bu"]
            self.w_pp_td = data["w_pp_td"]
            self.w_ip    = data["w_ip"]
            self.w_pi    = data["w_pi"]
        elif self.layer_type == LAYER_TYPE_BOTTOM:
            self.w_pp_bu = data["w_pp_bu"]
            self.w_pp_td = data["w_pp_td"]

        print("weight loaded: {}".format(real_file_path))
