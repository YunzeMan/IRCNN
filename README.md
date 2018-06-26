# IRCNN
Image Recovery Convolutional Neural Networks

This is a project of recovering damaged images. Specifically, the images that are damaged by adding noises.
A CNN with Residual blocks is used to tackle this challenge.

## Prerequisite
- tensorflow==1.4
- tensorlayer==1.8
- easydict
- CUDA8.0

## Train
To train your own model, please download your own dataset and change the path in `config.py`.
In my implementation, I a subset of VOC2012 as training data (about 14000 images) 


## Validate
Run the command in the project folder
```bash
python main.py mode==eval
```
