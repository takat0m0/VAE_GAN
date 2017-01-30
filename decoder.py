#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases, lrelu, deconv_layer
from batch_normalize import batch_norm


class Decoder(object):
    def __init__(self, z_dim, layer_chanels):
        self.z_dim = z_dim
        self.in_dim = 8
        self.layer_chanels = layer_chanels

        self.name_scope_reshape = u'dec_reshape_z'
        self.name_scope_deconv = u'dec_deconvolution'

    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_deconv in var.name or self.name_scope_reshape in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, z, batch_size, is_training):

        # reshape z
        with tf.variable_scope(self.name_scope_reshape):
            w_r = get_weights('_r',
                              [self.z_dim, self.in_dim * self.in_dim * self.layer_chanels[0]],
                              0.02)
            b_r = get_biases('_r',
                             [self.in_dim * self.in_dim * self.layer_chanels[0]],
                             0.0)
            h = tf.matmul(z, w_r) + b_r
            h = batch_norm(h, 'reshape', is_training)
            #h = tf.nn.relu(h)
            h = lrelu(h)
            
        h = tf.reshape(h, [-1, self.in_dim, self.in_dim, self.layer_chanels[0]])

        # deconvolution
        layer_num = len(self.layer_chanels) - 1
        with tf.variable_scope(self.name_scope_deconv):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):
                deconved = deconv_layer(inputs = h,
                                        out_shape = [batch_size, self.in_dim * 2 ** (i + 1), self.in_dim * 2 **(i + 1), out_chan],
                                        filter_width = 5, filter_hight = 5,
                                        stride = 2, l_id = i)
                if i == layer_num -1:
                    h = tf.nn.tanh(deconved)
                else:
                    bn_deconved = batch_norm(deconved, i, is_training)
                    #h = tf.nn.relu(bn_deconved)
                    h = lrelu(bn_deconved)

        return h
        
    
if __name__ == u'__main__':
    g = Decoder(512, [256, 128, 32, 3])
    z = tf.placeholder(tf.float32, [None, 512])
    g.set_model(z, 100, True)
