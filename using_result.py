#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    z_inputs = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
    dump_figs(np.asarray(fig_gen(z_inputs)), 'sample_result2')
