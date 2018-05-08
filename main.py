# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from fig_util import get_figs, dump_figs

if __name__ == u'__main__':

    # figs dir
    dir_name = u'figs'

    # parameter
    fig_size = 64
    gray_scale = False
    batch_size = 100
    pre_epoch_num = 10
    epoch_num = 100
    z_dim = 100
    repeat_dec_train = 8
    
    # make model
    print('-- make model --')
    model = Model(z_dim, batch_size, fig_size, gray_scale)
    model.set_model()
    
    # get_data
    print('-- get figs--')
    figs = get_figs(dir_name, fig_size, gray_scale)
    print('num figs = {}'.format(len(figs)))
    
    # training
    print('-- begin training --')
    num_one_epoch = len(figs) //batch_size

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(pre_epoch_num):

            print('** pre_epoch {} begin **'.format(epoch))
            obj_vae, obj_dec, obj_disc = 0.0, 0.0, 0.0
            for step in range(num_one_epoch):
                
                # get batch data
                batch_figs = figs[step * batch_size: (step + 1) * batch_size]
                batch_z = np.random.randn(batch_size, z_dim)
                # train
                obj_disc += model.pretraining_disc(sess, batch_figs, batch_z)
                obj_vae += model.pretraining_vae(sess, batch_figs)
                obj_dec += model.pretraining_dec(sess, batch_figs, batch_z)
                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    tmp_z = model.encoding(sess, batch_figs)
                    tmp_figs = model.gen_fig(sess, tmp_z)
                    #tmp_figs = model.gen_fig(sess, batch_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result')
                    
            print('epoch:{}, v_obj = {}, dec_obj = {}, disc_obj = {}'.format(epoch,
                                                                        obj_vae/num_one_epoch,
                                                            obj_dec/num_one_epoch,
                                                            obj_disc/num_one_epoch))
            saver.save(sess, './model.dump')
            
        for epoch in range(epoch_num):

            print('** epoch {} begin **'.format(epoch))
            obj_vae, obj_dec, obj_disc = 0.0, 0.0, 0.0
            for step in range(num_one_epoch):
                
                # get batch data
                batch_figs = figs[step * batch_size: (step + 1) * batch_size]
                batch_z = np.random.randn(batch_size, z_dim)
                # train
                obj_disc += model.training_disc(sess, batch_figs, batch_z)
                obj_vae += model.training_vae(sess, batch_figs)
                for _ in range(repeat_dec_train - 1):
                    model.training_dec(sess, batch_figs, batch_z)
                obj_dec += model.training_dec(sess, batch_figs, batch_z)

                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    tmp_z = model.encoding(sess, batch_figs)
                    tmp_figs = model.gen_fig(sess, tmp_z)
                    #tmp_figs = model.gen_fig(sess, batch_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result2')

                    
            print('epoch:{}, v_obj = {}, dec_obj = {}, disc_obj = {}'.format(epoch,
                                                                        obj_vae/num_one_epoch,
                                                            obj_dec/num_one_epoch,
                                                            obj_disc/num_one_epoch))
            saver.save(sess, './model.dump')
