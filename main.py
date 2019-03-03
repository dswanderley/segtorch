# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 10:39:30 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Main scripti with network definitons
"""

import os
import time
import torch
import torchvision

import numpy as np
import torch.nn as nn
import utils.transformations as tsfrm

from torch import optim
from utils.logger import Logger
from nets.unet import Unet2
from utils.datasets import UltrasoundDataset
from utils.losses import DiceLoss
from train import Training


# Get time to generate output name
def gettrainname(name):
    '''
    Get the train name with the training start full date.

    Arguments:
    @name (string): network name

    Returns: full_name (string)
    '''
    tm = time.gmtime()
    st_mon = str(tm.tm_mon) if tm.tm_mon > 9 else '0'+ str(tm.tm_mon)
    st_day = str(tm.tm_mday) if tm.tm_mday > 9 else '0'+ str(tm.tm_mday)
    st_hour = str(tm.tm_hour) if tm.tm_hour > 9 else '0'+ str(tm.tm_hour)
    st_min = str(tm.tm_min) if tm.tm_min > 9 else '0'+ str(tm.tm_min)
    tm_str = str(tm.tm_year) + st_mon + st_day + '_' + st_hour + st_min
    # log name - eg: both
    return tm_str + '_' + name


if __name__ == '__main__':

    # Define training name
    train_name = gettrainname('Unet2')

    # Set logs folder
    logger = Logger('./logs/' + train_name + '/')

    # Load Unet
    model = Unet2(n_channels=1, n_classes=[3,2])
    #print(net)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformation parameters
    transform = tsfrm.Compose([tsfrm.RandomHorizontalFlip(p=0.5),
                           tsfrm.RandomVerticalFlip(p=0.5),
                           tsfrm.RandomAffine(90, translate=(0.15, 0.15), scale=(0.75, 1.5), resample=3, fillcolor=0)
                           ])
    dataset_train = UltrasoundDataset(im_dir='Dataset/im/train/', gt_dir='Dataset/gt/train/', transform=transform)
    dataset_val = UltrasoundDataset(im_dir='Dataset/im/val/', gt_dir='Dataset/gt/val/')
    optmizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    loss_function = DiceLoss() # nn.CrossEntropyLoss()

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optmizer, loss_function, logger = logger, train_name=train_name)
    training.train(epochs=500, batch_size=3)
