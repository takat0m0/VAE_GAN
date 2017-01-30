#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2
from Model import Model
from util import get_figs, dump_figs

class FigGenerator(object):
    def __init__(self, file_name, z_dim, batch_size):

        self.batch_size = batch_size
        
        self.model = Model(z_dim, batch_size)
        self.model.set_model()
        saver = tf.train.Saver()
        self.sess = tf.Session()

        saver.restore(self.sess, file_name)
    def encoding(self, figs):
        return self.model.encoding(self.sess, figs)
    
    def __call__(self, z_inputs):
        assert(len(z_inputs) == self.batch_size)
        return self.model.gen_fig(self.sess, z_inputs)

if __name__ == u'__main__':

    # dump file
    dump_file = u'./model.dump'
    
    # parameter
    batch_size = 1
    z_dim = 100

    # figure generator
    fig_gen = FigGenerator(dump_file, z_dim, batch_size)

    # make figure
    fig1 = cv2.imread('./input1.jpg')
    fig1 = fig1/127.5 - 1.0
    fig2 = cv2.imread('./input2.jpg')
    fig2 = fig2/127.5 - 1.0
    
    z1 = fig_gen.encoding([fig1])[0]
    z2 = fig_gen.encoding([fig2])[0]
    diff = (z2 - z1)/50.0
    zs = []
    for i in range(51):
        z_target = z1 + diff * float(i)
        zs.append(fig_gen([z_target])[0])
    dump_figs(np.asarray(zs), 'morphing_result')
