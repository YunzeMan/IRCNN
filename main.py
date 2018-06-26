#! /usr/bin/python

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import IRCNN
from utils import *
from config import config, log_config

os.environ["CUDA_VISIBLE_DEVICES"]= '1'

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## learning IRCNN
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    logging.basicConfig(level=logging.DEBUG,
                        filename='train_info.log',
                        filemode='w',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    # create folders to save tested result images
    save_dir_ircnn = "samples/{}_ircnn".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ircnn)
    save_dir_ircnn_train = "samples/train_{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ircnn_train)
    checkpoint_dir = "checkpoint"  # checkpoint
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.jpg', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.jpg', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.jpg', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.jpg', printable=False))
    final_hr_img_list_4 = sorted(tl.files.load_file_list(path=config.FINAL.hr_img_path_4, regx='.*.png', printable=False))
    final_lr_img_list_4 = sorted(tl.files.load_file_list(path=config.FINAL.lr_img_path_4, regx='.*.png', printable=False))
    final_hr_img_list_6 = sorted(tl.files.load_file_list(path=config.FINAL.hr_img_path_6, regx='.*.png', printable=False))
    final_lr_img_list_6 = sorted(tl.files.load_file_list(path=config.FINAL.lr_img_path_6, regx='.*.png', printable=False))

    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=16)


    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, None, None, 3], name='t_image')
    t_target_image = tf.placeholder('float32', [None, None, None, 3], name='t_target_image')

    net = IRCNN(t_image, is_train=True, reuse=False)

    net.print_params(False)
    net.print_layers()

    ## test inference
    net_test = IRCNN(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    loss = tl.cost.mean_squared_error(net.outputs, t_target_image, is_mean=True)

    net_vars = tl.layers.get_variables_with_name('IRCNN', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## IRCNN
    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=net_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/{}.npz'.format(tl.global_flag['mode']), network=net) is True:
        print('Load Last Checkpoint to IRCNN Model.\n')
    else:
        print('There is no checkpoint, do not load the model.\n')
    

    ###============================= TRAINING ===============================###

    final_hr_imgs = tl.vis.read_images(final_hr_img_list_6, path=config.FINAL.hr_img_path_6, n_threads=16)
    sample_imgs_hr = tl.prepro.threading_data(final_hr_imgs, fn=normalize_img_noresize, is_random=False)
    print('final HR sub-image:', sample_imgs_hr.shape, sample_imgs_hr.min(), sample_imgs_hr.max())
    final_lr_imgs = tl.vis.read_images(final_lr_img_list_6, path=config.FINAL.lr_img_path_6, n_threads=16)
    sample_imgs_lr = tl.prepro.threading_data(final_lr_imgs, fn=normalize_img_noresize, is_random=False)
    print('final LR sub-image:', sample_imgs_lr.shape, sample_imgs_lr.min(), sample_imgs_lr.max())

    tl.vis.save_images(sample_imgs_lr, [1, 1], save_dir_ircnn + '/_final_sample_lr.jpg')
    tl.vis.save_images(sample_imgs_hr, [1, 1], save_dir_ircnn + '/_final_sample_hr.jpg')

    ### ========================= train IRCNN ========================= ###

    err_final, out = sess.run([loss,net_test.outputs], {t_image: sample_imgs_lr, t_target_image: sample_imgs_hr})  
    logging.debug("[*] Epoch: [%2d/%2d], Square Error on noise 0.6 is %f" %(0, n_epoch, err_final))
    print ("[*] save images")
    tl.vis.save_images(out, [1, 1], save_dir_ircnn + '/train_%d.png' % 0)


    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f " % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_hr = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=normalize_img, is_random=True)
            b_imgs_lr = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=normalize_img_add_noise, noiseRatio=0.6)
            ## update IRCNN
            err, out, _ = sess.run([loss, net.outputs, optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})

            print("Epoch [%2d/%2d] %4d time: %4.4fs, loss: %.8f" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, err))
            total_loss += err
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, 
                                                                                total_loss / n_iter)
        print(log)

        ## quick evaluation on final set
        if (epoch != 0) and (epoch % 5 == 0):
            err_final, out = sess.run([loss,net_test.outputs], {t_image: sample_imgs_lr, t_target_image: sample_imgs_hr})  
            logging.debug("[*] Epoch: [%2d/%2d], Square Error on noise 0.6 is %f" %(epoch, n_epoch, err_final))
            print("[*] save images")
            tl.vis.save_images(out, [1, 1], save_dir_ircnn + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 5 == 0):
            tl.files.save_npz(net.all_params, name=checkpoint_dir + '/{}.npz'.format(tl.global_flag['mode']), sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir_ircnn = "final_image"
    tl.files.exists_or_mkdir(save_dir_ircnn)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_lr_img_list = ["B.png"]
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path='.', n_threads=32)

    ###========================== DEFINE MODEL ============================###
    valid_lr_img = valid_lr_imgs[0]
    valid_lr_img = (valid_lr_img / 127.5) - 1 
    rols, cols, channels = valid_lr_img.shape
    valid_lr_img = np.reshape(valid_lr_img, (1, rols, cols, channels))
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net = IRCNN(t_image, is_train=False, reuse=False)

    ###===================== RESTORE IRCNN AND TEST =========================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/ircnn.npz', network=net)

    out = sess.run(net.outputs, {t_image: valid_lr_img})  
    print("[*] save images")
    tl.vis.save_images(out, [1, 1], save_dir_ircnn + '/final.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='ircnn', help='ircnn, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'ircnn':
        train()
    elif tl.global_flag['mode'] == 'eval':
        evaluate()
    else:
        raise Exception("Unknown --mode")
