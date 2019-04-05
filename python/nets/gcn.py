# -*- coding: utf-8 -*-
"""
Created on Fri Apr 05 10:10:10 2019

@author: Diego Wanderley
@python: 3.6
@description: GCN networks models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from nets.modules import *


class GCN(nn.Module):
    '''
        Fully Global Convolution Network
    '''
    def __init__(self, n_channels, n_classes):
        ''' Constructor '''
        super(GCN, self).__init__()
        # Number of classes definition
        self.n_classes = n_classes

        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = fwdconv(n_channels, 3, kernel_size=1)

        # Load Resnet
        resnet = models.resnet50(pretrained=True)

        # Set input layer
        self.conv0 = resnet.conv1 # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn0 = resnet.bn1 # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = resnet.relu
        self.maxpool0 = resnet.maxpool # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Resnet layers
        self.resnet1 = resnet.layer1 # res-2 -> (depth) in:   64, out:  256
        self.resnet2 = resnet.layer2 # res-3 -> (depth) in:  256, out:  512
        self.resnet3 = resnet.layer3 # res-4 -> (depth) in:  512, out: 1024
        self.resnet4 = resnet.layer4 # res-5 -> (depth) in: 1024, out: 2048

        # GCN Layers
        self.gcn1 = globalconv(256,  n_classes, k=55)
        self.gcn2 = globalconv(512,  n_classes, k=27)
        self.gcn3 = globalconv(1024, n_classes, k=13)
        self.gcn4 = globalconv(2048, n_classes, k=7)

        # Boundary Refine layers
        self.br1 = brconv(n_classes)
        self.br2 = brconv(n_classes)
        self.br3 = brconv(n_classes)
        self.br4 = brconv(n_classes)
        self.br5 = brconv(n_classes)
        self.br6 = brconv(n_classes)
        self.br7 = brconv(n_classes)
        self.br8 = brconv(n_classes)
        self.br9 = brconv(n_classes)

        # Deconv
        self.deconv1 = upconv(n_classes, n_classes) # Spatial size 16 x 16 -> 32 x 32
        self.deconv2 = upconv(n_classes, n_classes) # Spatial size 32 x 32 -> 64 x 64
        self.deconv3 = upconv(n_classes, n_classes) # Spatial size 64 x 64 -> 128x128
        self.deconv4 = upconv(n_classes, n_classes) # Spatial size 128x128 -> 256x256
        self.deconv5 = upconv(n_classes, n_classes) # Spatial size 256x256 -> 512x512

        # Softmax
        self.softmax = nn.Softmax2d()

        
    def forward(self, x):
        ''' Foward method '''

        # input (adapt to resnet input 3ch)
        c_x0 = self.inconv(x)

        # Resnet 
        dc_x0 = self.conv0(c_x0)    
        dc_x0 = self.bn0(dc_x0)
        dc_x0 = self.relu0(dc_x0)
        # downstream
        dc_x0 = self.maxpool0(dc_x0)    # 512x512 -> 256x256
        dc_x1 = self.resnet1(dc_x0)     # 256x256 -> 128x128
        dc_x2 = self.resnet2(dc_x1)     # 128x128 -> 64 x 64
        dc_x3 = self.resnet3(dc_x2)     # 64 x 64 -> 32 x 32
        dc_x4 = self.resnet4(dc_x3)     # 32 x 32 -> 16 x16

        # skip conections with global convs
        sc_x1 = self.gcn1(dc_x1)
        sc_x2 = self.gcn2(dc_x2)
        sc_x3 = self.gcn3(dc_x3)
        sc_x4 = self.gcn4(dc_x4)
        # 
        sc_x1 = self.br1(sc_x1)
        sc_x2 = self.br2(sc_x2)
        sc_x3 = self.br3(sc_x3)
        sc_x4 = self.br4(sc_x4)

        # upstream
        uc_x1   = self.conv_up1(sc_x4)
        uc_x1_s = sc_x3 + uc_x1
        uc_x1_r = self.br5(uc_x1_s)
        #
        uc_x2   = self.conv_up2(uc_x1_r)
        uc_x2_s = sc_x2 + uc_x2
        uc_x2_r = self.br6(uc_x2_s)
        #
        uc_x3   = self.conv_up3(uc_x2_r)
        uc_x3_s = sc_x1 + uc_x3
        uc_x3_r = self.br7(uc_x3_s)
        #
        uc_x4   = self.conv_up4(uc_x3_r)
        uc_x4_r = self.conv_up4(uc_x4)
        #
        uc_x5   = self.conv_up5(uc_x4_r)
        uc_x5_r = self.conv_up5(uc_x5)

        # output
        x_out = self.softmax(uc_x5_r)             

        return x
