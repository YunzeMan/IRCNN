from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## learning IRCNN
config.TRAIN.n_epoch = 100
config.TRAIN.lr_decay = 0.05
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '/home/manyz/dataset/IRCNN/IRCNN_train/hr_images/'
config.TRAIN.lr_img_path = '/home/manyz/dataset/IRCNN/IRCNN_train/lr_images/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '/home/manyz/dataset/IRCNN/IRCNN_test/hr_images/'
config.VALID.lr_img_path = '/home/manyz/dataset/IRCNN/IRCNN_test/lr_images/'


config.FINAL = edict()
# finally needed images
config.FINAL.hr_img_path_4 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_4/hr_images/'
config.FINAL.lr_img_path_4 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_4/lr_images/'
config.FINAL.hr_img_path_6 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_6/hr_images/'
config.FINAL.lr_img_path_6 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_6/lr_images/'
config.FINAL.hr_img_path_8 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_8/hr_images/'
config.FINAL.lr_img_path_8 = '/home/manyz/dataset/IRCNN/IRCNN_final/noise_8/lr_images/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
