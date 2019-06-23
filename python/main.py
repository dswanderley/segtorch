# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 10:39:30 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Main scripizt with network definitons
"""

import os
import ast
import sys
import time
import argparse
import torch
import torchvision

import torch.nn as nn
import utils.transformations as tsfrm

from torch import optim
from utils.logger import Logger
from nets.deeplab import DeepLabv3, DeepLabv3_plus
from nets.unet import *
from nets.dilation import *
from nets.gcn import *
from nets.fcn import *
from utils.datasets import OvaryDataset, VOC2012Dataset
from utils.losses import *
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

    parser = argparse.ArgumentParser(description="PyTorch segmentation network training and prediction.")
    parser.add_argument('--net', type=str, default='unet2',
                        choices=['fcn_r101', 'fcn_r50',
                                'deeplabv3', 'deeplabv3_r50', 'deeplabv3p', 'deeplabv3p_r50',
                                 'unet', 'unet_light', 'unet2', 'd_unet2',
                                 'sp_unet', 'sp_unet2',
                                 'gcn', 'gcn2', 'b_gcn', 'u_gcn'],
                        help='network name (default: unet2)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size (default: 1)')
    parser.add_argument('--dataset', type=str, default='ovarian',
                        choices=['ovarian', 'voc2012'],
                        help='select dataset, it also defines the input depth and the output classes (default: ovarian)')
    parser.add_argument('--multitask', type=str, default='none',
                        choices=['none', 'ovary', 'follicle', 'both'],
                        help='select dataset (default: ovarian)')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=['adam', 'adamax', 'sgd'],
                        help='optmization process (default: adam)')
    parser.add_argument('--loss', type=str, default='dsc',
                        choices=['dice', 'wdice', 'discriminative', 'crossentropy'],
                        help='loss function (default: dice)')
    parser.add_argument('--clahe', type=bool, default=False,
                        help='whether to use adaptive histogram equalization (default: False)')
    parser.add_argument('--interaction', type=bool, default=False,
                        help='whether to use interaction points (default: False)')
    parser.add_argument('--bilinear', type=bool, default=False,
                        help='whether to use bilinear upsampling should be used instead of Transpose Conv. (default: False)')

    # Parse input data
    args = parser.parse_args()

    # Input parameters
    n_epochs = args.epochs
    batch_size = args.batch_size
    dataset_name = args.dataset
    opt = args.opt
    loss = args.loss
    net_type = args.net
    bilinear = args.bilinear
    clahe = args.clahe
    interaction = args.interaction
    multitask = args.multitask

    network_name = net_type

    # Manage network input - Ovarian dataset parameters
    if dataset_name == 'ovarian':
        in_channels = 1
        # Multitask
        if multitask == 'ovarian':
            n_classes = [3,2]
            target = ['gt_mask', 'ovary_mask']
            network_name += '_ovar'
        elif multitask == 'follicle':
            n_classes = [3,2]
            target = ['gt_mask', 'follicle_edge']
            network_name += '_foll'
        elif multitask == 'both':
            n_classes = [3,2,2]
            target = ['gt_mask', 'ovary_mask', 'follicle_edge']
            network_name += '_both'
        else:
            n_classes = 3
            target = 'gt_mask'
    # VOC 2012
    else:
        in_channels = 3
        n_classes = 22
        target = 'gt_mask'
        network_name += '_voc2012'

    if clahe:
        in_channels += 1

    if interaction:
        interaction = [1., 0.5]
        in_channels += 1
        network_name = 'i' + network_name

    print('--- Network Parameters ---')
    print('network name:   {:s}'.format(network_name))
    print('dataset:        {:s}'.format(dataset_name))
    print('output classes: {:d}'.format(n_classes))
    print('epochs:         {:d}'.format(n_epochs))
    print('batch size:     {:d}'.format(batch_size))
    print('optmization:    {:s}'.format(opt))
    print('loss funcion:   {:s}'.format(loss))

    # Define training name
    train_name = gettrainname(network_name)

    # Load Network model
    if net_type == 'fcn_r101':
        model = FCN(n_channels=in_channels, n_classes=n_classes, resnet_type=101, pretrained=True)
    elif net_type == 'fcn_r50':
        model = FCN(n_channels=in_channels, n_classes=n_classes, resnet_type=50)
    # Deeplab v3
    elif net_type == 'deeplabv3':
        model = DeepLabv3(n_channels=in_channels, n_classes=n_classes, resnet_type=101, pretrained=True)
    elif net_type == 'deeplabv3_r50':
        model = DeepLabv3(n_channels=in_channels, n_classes=n_classes, resnet_type=50, pretrained=True)
    # Deeplab v3+
    elif net_type == 'deeplabv3p':
        model = DeepLabv3_plus(nInputChannels=in_channels, n_classes=n_classes, os=16, pretrained=True)
    elif net_type == 'deeplabv3p_r50':
        model = DeepLabv3_plus(nInputChannels=in_channels, n_classes=n_classes, os=16, resnet_type=50, pretrained=True)
    # Global Convolution Network
    elif net_type == 'gcn':
        model = GCN(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'b_gcn':
        model = BalancedGCN(n_channels=in_channels, n_classes=n_classes)
    # Unet models
    elif net_type == 'unet':
        model = Unet(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'unet_light':
        model = UnetLight(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'sp_unet':
        model = SpatialPyramidUnet(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'sp_unet2':
        model = SpatialPyramidUnet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'd_unet':
        model = DilatedUnet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    else:
        model = Unet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)

    print('model class:   {:s}'.format(str(type(model))))
    print('--------------------------')
    print('')

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformation parameters
    transform = tsfrm.Compose([tsfrm.RandomHorizontalFlip(p=0.5),
                           tsfrm.RandomVerticalFlip(p=0.5),
                           tsfrm.RandomAffine(90, translate=(0.15, 0.15), scale=(0.75, 1.5), resample=3, fillcolor=0)
                           ])

    # Dataset definitions
    if dataset_name == 'ovarian':
        im_dir = '../datasets/ovarian/im/'
        gt_dir = '../datasets/ovarian/gt/'
        dataset_train = OvaryDataset(im_dir=im_dir+'train/',gt_dir=gt_dir+'train/',  imap=interaction, clahe=clahe, transform=transform)
        dataset_val =   OvaryDataset(im_dir=im_dir+'val/',  gt_dir=gt_dir+'val/',    imap=interaction, clahe=clahe)
        dataset_test =  OvaryDataset(im_dir=im_dir+'test/', gt_dir=gt_dir+'test/',   imap=interaction, clahe=clahe)
    else:
        im_dir = '../datasets/voc2012/JPEGImages/'
        gt_dir = '../datasets/voc2012/SegmentationClass/'
        list_dir = '../datasets/voc2012/'
        dataset_train = VOC2012Dataset(im_dir=im_dir, gt_dir=gt_dir, file_list=list_dir+'train.txt')
        dataset_val =   VOC2012Dataset(im_dir=im_dir, gt_dir=gt_dir, file_list=list_dir+'val.txt')
        dataset_test =  VOC2012Dataset(im_dir=im_dir, gt_dir=gt_dir, file_list=list_dir+'val.txt')

    # Training Parameters
    if opt == 'adam':
        optmizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt == 'adamax':
        optmizer = optim.Adamax(model.parameters(), lr=0.002)
        train_name += '_adamax'
    else:
        optmizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=0.0005)
        train_name += '_sdg'

    # Loss function
    if loss == 'dsc' or loss == 'dice':
        loss_function = DiceLoss()
    elif loss == 'wdice' or loss == 'weighted_dice':
        loss_function = WeightedDiceLoss()
        train_name += '_wdsc'
    elif loss == 'discriminative' or loss == 'dlf':
        loss_function = DiscriminativeLoss(n_features=2)
        train_name += '_dlf'
    else:
        loss_function = nn.CrossEntropyLoss()
        train_name += '_entp'

    # Set logs folder
    logger = Logger('../logs/' + train_name + '/')

    # Run training
    training = Training(model, device, dataset_train, dataset_val,
                        optmizer, loss_function, target=target,
                        logger=logger, train_name=train_name, arch=net_type)
    training.train(epochs=n_epochs, batch_size=batch_size)
    print('------------- END OF TRAINING -------------')
    print(' ')

    # Test network model
    print('Testing')
    print('')
    weights_path = '../weights/' + train_name + '_weights.pth.tar'
    # Output folder
    out_folder = '../predictions/' + train_name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # Load inference
    inference = Inference(model, device, weights_path, folder=out_folder)
    inference.predict(dataset_test)