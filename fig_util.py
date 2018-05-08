# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import cv2

def get_figs(dir_name, fig_size, gray_scale = False):
    ret = []
    for file_name in os.listdir(dir_name):
        tmp = cv2.imread(os.path.join(dir_name, file_name))
        if gray_scale:
            tmp = cv2.resize(tmp, (fig_size, fig_size))
            tmp = np.reshape(tmp, (fig_size, fig_size, 1))
        else:
            tmp = cv2.resize(tmp, (fig_size, fig_size))
        ret.append(tmp/127.5 - 1.0)
    return np.asarray(ret, dtype = np.float32)

def dump_figs(imgs, dir_name):
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dir_name, '{}.jpg'.format(i)), (img + 1.0) * 127.5)
