# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from discriminator import Discriminator

class Model(object):
    def __init__(self, z_dim, batch_size, fig_size, gray_scale = False):
        self.z_dim = z_dim
        self.batch_size = batch_size
        
        self.lr = 0.0001
        self.gamma = 0.5

        if gray_scale:
            self.x = tf.placeholder(tf.float32, [self.batch_size, fig_size, fig_size, 1])
        else:
            self.x = tf.placeholder(tf.float32, [self.batch_size, fig_size, fig_size, 3])
        
        # -- encoder -------
        self.enc = Encoder([3, 64, 128, 256], 2048, z_dim)
        
        # -- decoder -------
        self.dec = Decoder(z_dim, [256, 128, 32, 3])

        # -- discriminator --
        self.disc = Discriminator([3, 32, 128, 256, 256], 512)
        
        
    def set_model(self):

        
        # -- VAE ---------
        mu, log_sigma = self.enc.set_model(self.x, is_training = True)
        obj_kl = tf.reduce_sum(mu * mu/2.0 - log_sigma + tf.exp(2.0 * log_sigma)/2.0 - 0.5, 1)
        obj_kl = tf.reduce_mean(obj_kl, 0)
                
        eps = tf.random_normal([self.batch_size, self.z_dim])
        z = eps * tf.exp(log_sigma) + mu

        vae_gen_figs = self.dec.set_model(z, self.batch_size, is_training = True)
        vae_logits, vae_feature_image = self.disc.set_model(vae_gen_figs, is_training = True)
        reconstruct_error = tf.reduce_mean(
            tf.reduce_sum(tf.pow(vae_gen_figs - self.x, 2), [1, 2, 3]))

        obj_dec_from_vae = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = vae_logits,
                labels = tf.ones_like(vae_logits)))
        
        obj_disc_from_vae = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = vae_logits,
                labels = tf.zeros_like(vae_logits)))

        # -- draw from prior -------
        self.z_pr = tf.placeholder(dtype = tf.float32, shape = [self.batch_size, self.z_dim])
        dec_figs = self.dec.set_model(self.z_pr, self.batch_size, is_training = True, reuse = True)
        dec_logits, _ = self.disc.set_model(dec_figs, is_training = True, reuse = True)

        obj_dec_from_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = dec_logits,
                labels = tf.ones_like(dec_logits)))

        obj_disc_from_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = dec_logits,
                labels = tf.zeros_like(dec_logits)))

        # -- obj from inputs --------
        disc_logits, input_feature_image = self.disc.set_model(self.x, is_training = True, reuse = True)
        obj_disc_from_inputs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = disc_logits,
                labels = tf.ones_like(disc_logits)))
        u'''
        dis_similar = tf.reduce_mean(
            tf.reduce_sum(pow(tf.nn.sigmoid(vae_logits) -
                              tf.nn.sigmoid(disc_logits), 2), 1))
        '''
        dis_similar = tf.reduce_mean(
            tf.reduce_sum(pow(vae_feature_image - input_feature_image, 2), [1, 2, 3]))
        # == setting obj ============
        # -- pretrain --------
        self.pre_obj_vae = reconstruct_error + obj_kl
        train_vars = self.enc.get_variables()
        train_vars.extend(self.dec.get_variables())
        self.pretrain_vae  = tf.train.AdamOptimizer(self.lr)\
                                     .minimize(self.pre_obj_vae,
                                               var_list = train_vars)


        self.pre_obj_dec = obj_dec_from_prior
        train_vars = self.dec.get_variables()
        self.pretrain_dec  = tf.train.AdamOptimizer(self.lr).minimize(self.pre_obj_dec, var_list = train_vars)

        
        self.pre_obj_disc = obj_disc_from_prior + obj_disc_from_inputs
        train_vars = self.disc.get_variables()
        self.pretrain_disc  = tf.train.AdamOptimizer(self.lr).minimize(self.pre_obj_disc, var_list = train_vars)

        # -- train -----
        self.obj_vae = dis_similar + obj_kl
        train_vars = self.enc.get_variables()
        self.train_vae  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_vae, var_list = train_vars)

        self.obj_dec = obj_dec_from_vae + obj_dec_from_prior + self.gamma * dis_similar
        train_vars = self.dec.get_variables()
        self.train_dec = tf.train.AdamOptimizer(self.lr).minimize(self.obj_dec, var_list = train_vars)

        self.obj_disc = obj_disc_from_vae + obj_disc_from_prior + obj_disc_from_inputs
        train_vars = self.disc.get_variables()
        self.train_disc  = tf.train.AdamOptimizer(self.lr).minimize(self.obj_disc, var_list = train_vars)
        
        # -- for using ---------------------
        self.mu, _  = self.enc.set_model(self.x, is_training = False, reuse = True)
        self.dec_figs = self.dec.set_model(self.z_pr, self.batch_size, is_training = False, reuse = True)
        
    def pretraining_vae(self, sess, figs):
        _, pre_obj_vae = sess.run([self.pretrain_vae, self.pre_obj_vae],
                                  feed_dict = {self.x: figs})
        return pre_obj_vae
        
    def pretraining_dec(self, sess, figs, z):
        _, pre_obj_dec = sess.run([self.pretrain_dec, self.pre_obj_dec],
                                  feed_dict = {self.x: figs, self.z_pr:z})
        return pre_obj_dec
    
    def pretraining_disc(self, sess, figs, z):
        _, pre_obj_disc = sess.run([self.pretrain_disc, self.pre_obj_disc],
                                  feed_dict = {self.x: figs, self.z_pr:z})
        return pre_obj_disc
    
    def training_vae(self, sess, figs):
        _, obj_vae = sess.run([self.train_vae, self.obj_vae],
                                  feed_dict = {self.x: figs})
        return obj_vae
        
    def training_dec(self, sess, figs, z):
        _, obj_dec = sess.run([self.train_dec, self.obj_dec],
                                  feed_dict = {self.x: figs,
                                               self.z_pr:z})
        return obj_dec
    
    def training_disc(self, sess, figs, z):
        _, obj_disc = sess.run([self.train_disc, self.obj_disc],
                                  feed_dict = {self.x: figs,
                                               self.z_pr:z})
        return obj_disc
    
    def encoding(self, sess, figs):
        ret = sess.run(self.mu, feed_dict = {self.x: figs})
        return ret
    
    def gen_fig(self, sess, z):
        ret = sess.run(self.dec_figs,
                       feed_dict = {self.z_pr: z})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 100, batch_size = 100, fig_size = 64, gray_scale = False)
    model.set_model()
    
