# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 10:39:30 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Main scripizt with network definitons
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
from utils.datasets import OvaryDataset
from utils.losses import DiceLoss, DiscriminativeLoss
from train import Training
from predict import Inference


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

# Main calls
if __name__ == '__main__':

    # Input parameters
    network_name = 'Unet2'
    opt = 'adam'
    loss = 'dsc'
    batch_size = 3
    n_epochs = 1
    
    # Define training name
    train_name = gettrainname(network_name)

    # Set logs folder
    logger = Logger('./logs/' + train_name + '/')

    # Load Unet
    model = Unet2(n_channels=1, n_classes=3)
    #print(net)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformation parameters
    transform = tsfrm.Compose([tsfrm.RandomHorizontalFlip(p=0.5),
                           tsfrm.RandomVerticalFlip(p=0.5),
                           tsfrm.RandomAffine(90, translate=(0.15, 0.15), scale=(0.75, 1.5), resample=3, fillcolor=0)
                           ])
    # Dataset definitions
    dataset_train = OvaryDataset(im_dir='dataset/im/train/', gt_dir='dataset/gt/train/', transform=transform)
    dataset_val = OvaryDataset(im_dir='dataset/im/val/', gt_dir='dataset/gt/val/')
    dataset_test = OvaryDataset(im_dir='dataset/im/test/', gt_dir='dataset/gt/test/')

    # Training Parameters
    if opt == 'adam':
        optmizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optmizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=0.0005)
    # Loss function
    if loss == 'dsc' or loss == 'dice':
        loss_function = DiceLoss() 
    elif loss == 'discriminative' or loss == 'dlf':
        loss_function = DiscriminativeLoss() 
    else:
        loss_function = nn.CrossEntropyLoss()

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optmizer, loss_function, logger=logger, train_name=train_name)
    training.train(epochs=n_epochs, batch_size=batch_size)

    # Test network model
    weights_path = './weights/' + train_name + '_weights.pth.tar'
    inference = Inference(model, device, weights_path)
    inference.predict(dataset_test)