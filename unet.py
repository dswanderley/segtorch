# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 11:33:30 2018

@author: Diego Wanderley
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class inconv(nn.Module):
    '''
    Input layer
    '''
    def __init__(self, in_ch, out_ch):
        ''' Constructor '''
        super(inconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class fwdconv(nn.Module):
    '''
    Foward convolution layer
    '''
    def __init__(self, in_ch, out_ch):
        ''' Constructor '''
        super(fwdconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class downconv(nn.Module):
    '''
    Downconvolution layer
    '''
    def __init__(self, in_ch, out_ch):
        ''' Constructor '''
        super(downconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class upconv(nn.Module):
    '''
    Upconvolution layer
    '''
    def __init__(self, in_ch, out_ch, bilinear=False):
        ''' Constructor '''
        super(upconv, self).__init__()
        # Check interpolation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # Set conv layer
        self.conv = fwdconv(in_ch, out_ch)
        
    def forward(self, x, x_res):
        ''' Foward method '''
        x = torch.cat((x, x_res), 1)
        return x


class outconv(nn.Module):
    '''
    Output convolution layer
    '''
    def __init__(self, in_ch, out_ch):
        ''' Constructor '''
        super(outconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x



class Unet(nn.Module):
    '''
    U-net class
    '''
    def __init__(self, n_channels, n_classes):
        ''' Constructor '''
        super(Unet, self).__init__()
        # Set input layer
        self.conv_init  = inconv(n_channels, 8)
        # Set downconvolution layer 1
        self.conv_down1 = downconv(8, 16)
        # Set downconvolution layer 2
        self.conv_down2 = downconv(16, 32)
        # Set downconvolution layer 3
        self.conv_down3 = downconv(32, 64)
        # Set upconvolution layer 1
        self.conv_up1 = upconv(64, 32)
        # Set upconvolution layer 2
        self.conv_up2 = upconv(32, 16)
        # Set upconvolution layer 3
        self.conv_up3 = upconv(16, 8)
        # Set output layer
        self.conv_out = outconv(8, n_classes)
        
    def forward(self, x):
        ''' Foward method '''
        # input
        x0 = self.init_conv(x)
        # downstream
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)
        # upstream
        x4 = self.conv_up1(x3)
        x5 = self.conv_up2(x4)
        x6 = self.conv_up3(x5)
        # output
        x = self.conv_out(x6)
        return  F.sigmoid(x)

