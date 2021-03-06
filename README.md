# IRCNN-Tensorflow
Image Recovery Convolutional Neural Networks


- [IRCNN-Tensorflow](#ircnn-tensorflow)
    - [1. Introduction](#1-introduction)
        - [1.1 Overall Structure](#11-overall-structure)
        - [1.2 What kind of noise](#12-what-kind-of-noise-is-added)
    - [2. Prerequisites](#2-prerequisites)
    - [3. Demo](#3-demo)
    - [4. Train](#4-train)
    - [5. Test](#5-test)

## 1. Introduction
This is a project of recovering damaged images. Specifically, the images that are damaged by adding noises. Based on the testing result, this method is able to infer and recover the real image from the masked ones.
#### 1.1 Overall Structure
 A CNN with Residual blocks is used to tackle this challenge. It has a similar structure as [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092), namely SRCNN. Detailed structure is clearly shown in `model.py`.

#### 1.2 What kind of noise is added
Gaussian random noise of different ratio is added over 3 channels of the images, separately. The noise funtion is specified in `utils.py`, in function `normalize_img_add_noise(Img, noiseRatio)`.

Besides, you are welcomed to implement other kind of noise and test on them. 

## 2. Prerequisites
- tensorflow==1.4.1
- tensorlayer==1.8
- easydict
- CUDA8.0

## 3. Demo
**Noisy Image**


![Noisy](https://github.com/YunzeMan/IRCNN/blob/master/demo/B.png)

**Recovered Image**


![Recover](https://github.com/YunzeMan/IRCNN/blob/master/final_image/final.png)

## 4. Train
To train your own model, please download your own dataset and change the path in `config.py`. You may need to make some minor changes if you use dataset of different image size. 

Then simply run
```bash
python main.py
```
In this implementation, I use a subset of VOC2012 as training data (about 14000 images) 


## 5. Validate
To validate the demo image, run the command in the project folder
```bash
python main.py mode==eval
```
If you want to test on your own image, change the path in `main.py`