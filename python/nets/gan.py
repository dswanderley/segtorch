# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 17:13:05 2019

@author: Diego Wanderley
@python: 3.6
@description: Generative Adversarial Networks for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.modules import *
from nets.unet import *


class Discriminator(nn.Module):
    '''
    GAN Descriminator module
    '''
    def __init__(self, n_channels, n_classes, bilinear=False):
        ''' Constructor '''
        super(Discriminator, self).__init__()

        # Number of classes definition
        self.n_classes = n_classes

        # Set input layer
        self.conv_init  = inconv(n_channels, 8)

        # Set downconvolution layer 1
        self.conv_down1 = downconv(8, 8, dropout=0.2)
        # Set downconvolution layer 2
        self.conv_down2 = downconv(8, 16, dropout=0.2)
        # Set downconvolution layer 3
        self.conv_down3 = downconv(16, 24, dropout=0.2)
        # Set downconvolution layer 4
        self.conv_down4 = downconv(24, 32, dropout=0.2)
        # Set downconvolution layer 5
        self.conv_down5 = downconv(32, 40, dropout=0.2)
        # Set downconvolution layer 6
        self.conv_down6 = downconv(40, 48, dropout=0.2)


    def forward(self, x):
        ''' Foward method '''

        # input
        c_x0 = self.conv_init(x)
        # downstream
        dc_x1 = self.conv_down1(c_x0)
        dc_x2 = self.conv_down2(dc_x1)
        dc_x3 = self.conv_down3(dc_x2)
        dc_x4 = self.conv_down4(dc_x3)
        dc_x5 = self.conv_down5(dc_x4)
        dc_x6 = self.conv_down6(dc_x5)

        x_out = dc_x6

        return x_out


class GanSeg(nn.Module):
    '''
    U-net class from end-to-end ovarian structures segmentation
    '''
    def __init__(self, n_channels, n_classes, bilinear=False):
        ''' Constructor '''
        super(GanSeg, self).__init__()

        # Number of classes definition
        self.n_features = n_classes
        # Unet
        self.generator = Unet2(n_channels, 8)
        # Output
        self.discriminator = Discriminator(n_channels, n_classes)

    def forward(self, x):
        ''' Foward method '''
        # Unet
        x_1 = self.generator(x)
        # Output
        x_out = self.discriminator(x_1)

        return x_out

