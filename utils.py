import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import scipy
import numpy as np


def normalize_img(x, is_random=True):
    x = imresize(x, size=[128, 128], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def normalize_img_noresize(x, is_random=True):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def normalize_img_add_noise(Img, noiseRatio):
    Img = imresize(Img, size=[128, 128], interp='bicubic', mode=None)
    rows, cols, channels = Img.shape
    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = round(noiseRatio * cols)
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    corrImg = Img * noiseMask
    corrImg = corrImg / (255. / 2.)
    corrImg = corrImg - 1.
    return corrImg
