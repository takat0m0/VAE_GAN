#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases, get_dim, lrelu, conv_layer
from batch_normalize import batch_norm

class Encoder(object):
    def __init__(self, layer_chanels, fc_dim, z_dim):
        self.layer_chanels = layer_chanels
        self.fc_dim = fc_dim
        self.z_dim = z_dim
        
        self.name_scope_conv = u'enc_conv'
        self.name_scope_fc = u'enc_fc'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_conv in var.name or self.name_scope_fc in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, figs, is_training):

        u'''
        return only logits. not sigmoid(logits).
        '''
        
        h = figs
        
        # convolution
        with tf.variable_scope(self.name_scope_conv):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):

                conved = conv_layer(inputs = h,
                                    out_num = out_chan,
                                    filter_width = 5, filter_hight = 5,
                                    stride = 2, l_id = i)
                
                if i == 0:
                    h = tf.nn.relu(conved)
                    #h = lrelu(conved)
                else:
                    bn_conved = batch_norm(conved, i, is_training)
                    h = tf.nn.relu(bn_conved)
                    #h = lrelu(bn_conved)
        # full connect
        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])
        
        with tf.variable_scope(self.name_scope_fc):
            weights = get_weights('fc', [dim, self.fc_dim], 0.02)
            biases  = get_biases('fc', [self.fc_dim], 0.0)
            h = tf.matmul(h, weights) + biases
            h = batch_norm(h, 'en_fc_bn', is_training)
            h = tf.nn.relu(h)
            
            weights = get_weights('mu', [self.fc_dim, self.z_dim], 0.02)
            biases  = get_biases('mu', [self.z_dim], 0.0)
            mu = tf.matmul(h, weights) + biases
            
            weights = get_weights('sigma', [self.fc_dim, self.z_dim], 0.02)
            biases  = get_biases('sigma', [self.z_dim], 0.0)
            sigma = tf.exp(tf.matmul(h, weights) + biases)
            
        return mu, sigma
    
if __name__ == u'__main__':
    g = Encoder([3, 64, 128, 256], 2048, 512)
    figs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    g.set_model(figs, True)
