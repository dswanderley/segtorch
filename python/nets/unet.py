# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 11:33:30 2018

@author: Diego Wanderley
@python: 3.6
@description: U-net networks models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.modules import *


class Unet(nn.Module):
    '''
    U-net class
    '''
    def __init__(self, n_channels, n_classes):
        ''' Constructor '''
        super(Unet, self).__init__()
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

        # Set upconvolution layer 1
        self.conv_up1 = upconv(48, 320, res_ch=40, dropout=0.2)
        # Set upconvolution layer 2
        self.conv_up2 = upconv(320, 256, res_ch=32, dropout=0.2)
        # Set upconvolution layer 3
        self.conv_up3 = upconv(256, 192, res_ch=24, dropout=0.2)
        # Set upconvolution layer 4
        self.conv_up4 = upconv(192, 128, res_ch=16, dropout=0.2)
        # Set upconvolution layer 5
        self.conv_up5 = upconv(128, 64, res_ch=8, dropout=0.2)
        # Set upconvolution layer 6
        self.conv_up6 = upconv(64, 8, res_ch=8, dropout=0.2)

        # Output
        self.conv_out = outconv(8, n_classes)

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
        # upstream
        uc_x1 = self.conv_up1(dc_x6, dc_x5)
        uc_x2 = self.conv_up2(uc_x1, dc_x4)
        uc_x3 = self.conv_up3(uc_x2, dc_x3)
        uc_x4 = self.conv_up4(uc_x3, dc_x2)
        uc_x5 = self.conv_up5(uc_x4, dc_x1)
        uc_x6 = self.conv_up6(uc_x5, c_x0)
        # output
        x = self.conv_out(uc_x6)
        return x


class Unet2(nn.Module):
    '''
    U-net class from end-to-end ovarian structures segmentation
    '''
    def __init__(self, n_channels, n_classes, bilinear=False):
        ''' Constructor '''
        super(Unet2, self).__init__()

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

        # Set upconvolution layer 1
        self.conv_up1 = upconv(48, 320, res_ch=40, dropout=0.2, bilinear=bilinear)
        # Set upconvolution layer 2
        self.conv_up2 = upconv(320, 256, res_ch=32, dropout=0.2, bilinear=bilinear)
        # Set upconvolution layer 3
        self.conv_up3 = upconv(256, 192, res_ch=24, dropout=0.2, bilinear=bilinear)
        # Set upconvolution layer 4
        self.conv_up4 = upconv(192, 128, res_ch=16, dropout=0.2, bilinear=bilinear)
        # Set upconvolution layer 5
        self.conv_up5 = upconv(128, 64, res_ch=8, dropout=0.2, bilinear=bilinear)
        # Set upconvolution layer 6
        self.conv_up6 = upconv(64, 8, res_ch=8, dropout=0.2, bilinear=bilinear)

        # Set output layer
        if type(n_classes) is list:
            self.conv_out = nn.ModuleList() # necessary for GPU convertion
            for n in n_classes:
                c_out = outconv(8, n)
                self.conv_out.append(c_out)
        else:
            self.conv_out = outconv(8, n_classes)
        # Define Softmax
        self.softmax = nn.Softmax2d()

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
        # upstream
        uc_x1 = self.conv_up1(dc_x6, dc_x5)
        uc_x2 = self.conv_up2(uc_x1, dc_x4)
        uc_x3 = self.conv_up3(uc_x2, dc_x3)
        uc_x4 = self.conv_up4(uc_x3, dc_x2)
        uc_x5 = self.conv_up5(uc_x4, dc_x1)
        uc_x6 = self.conv_up6(uc_x5, c_x0)
        # output
        if type(self.n_classes) is list:
            x_out = []
            for c_out in self.conv_out:
                x = c_out(uc_x6)
                x_out.append(self.softmax(x))
        else:
            x = self.conv_out(uc_x6)
            x_out = self.softmax(x)             

        return x_out


class InstSegNet(nn.Module):

    def __init__(self, n_channels, n_features):
        ''' Constructor '''
        super(InstSegNet, self).__init__()

        # Number of classes definition
        self.n_features = n_features
        # Unet
        self.body = Unet(n_channels, 8)
        # Output
        self.conv_out = outconv(8, n_features)

    def forward(self, x):
        ''' Foward method '''

        # Unet
        x_1 = self.body(x)
        # Output
        x_out = self.conv_out(x_1)

        return x_out