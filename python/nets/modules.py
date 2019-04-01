# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 17:19:35 2019

@author: Diego Wanderley
@python: 3.6
@description: CNN modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class inconv(nn.Module):
    '''
    Input layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(inconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1))
        if batch_norm:
            self.conv.add_module("bnorm_1", nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout_1", nn.Dropout2d(dropout))
        self.conv.add_module("relu_1", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class fwdconv(nn.Module):
    '''
    Foward convolution layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(fwdconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1))
        if batch_norm:
            self.conv.add_module("bnorm_1", nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout_1", nn.Dropout2d(dropout))
        self.conv.add_module("relu_1", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class downconv(nn.Module):
    '''
    Downconvolution layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(downconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
        if batch_norm:
            self.conv.add_module("bnorm_1",nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout_1", nn.Dropout2d(dropout))
        self.conv.add_module("relu_1",nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class upconv(nn.Module):
    '''
    Upconvolution layer
    '''
    def __init__(self, in_ch, out_ch, res_ch=0, bilinear=False, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(upconv, self).__init__()
        # Check interpolation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("fwdconv_1", fwdconv(in_ch+res_ch, out_ch, batch_norm=True, dropout=0))
        self.conv.add_module("fwdconv_2", fwdconv(out_ch, out_ch))

    def forward(self, x, x_res=None):
        ''' Foward method '''
        x_up = self.up(x)

        if x_res is None:
            x_cat = x_up
        else:
            x_cat = torch.cat((x_up, x_res), 1)

        x_conv = self.conv(x_cat)

        return x_conv


class outconv(nn.Module):
    '''
    Output convolution layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(outconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0))
        if batch_norm:
            self.conv.add_module("bnorm_1",nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout_1", nn.Dropout2d(dropout))
        self.conv.add_module("relu_1",nn.ReLU(inplace=True))
              

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x
