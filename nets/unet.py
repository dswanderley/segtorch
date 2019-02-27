# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 11:33:30 2018

@author: Diego Wanderley
@python: 3.6
@description: U-net modules and network
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
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
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
    def __init__(self, in_ch, out_ch, res_ch=0, bilinear=False):
        ''' Constructor '''
        super(upconv, self).__init__()
        # Check interpolation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # Set conv layer
        self.conv = fwdconv(in_ch+res_ch, out_ch)
        
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
    def __init__(self, in_ch, out_ch):
        ''' Constructor '''
        super(outconv, self).__init__()
        # Set conv layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        x = self.softmax(x)
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
        self.conv_up1 = upconv(64, 32, res_ch=32)
        # Set upconvolution layer 2
        self.conv_up2 = upconv(32, 16, res_ch=16)
        # Set upconvolution layer 3
        self.conv_up3 = upconv(16, 8, res_ch=8)
        # Set output layer
        self.conv_out = outconv(8, n_classes)
        
    def forward(self, x):
        ''' Foward method '''
        # input
        x0 = self.conv_init(x)
        # downstream
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)
        # upstream
        x4 = self.conv_up1(x3, x2)
        x5 = self.conv_up2(x4, x1)
        x6 = self.conv_up3(x5, x0)
        # output
        x = self.conv_out(x6)
        return x
    

class Unet2(nn.Module):
    '''
    U-net class from end-to-end ovarian structures segmentation
    '''
    def __init__(self, n_channels, n_classes):
        ''' Constructor '''
        super(Unet2, self).__init__()

        # Set input layer
        self.conv_init  = inconv(n_channels, 8)

        # Set downconvolution layer 1
        self.conv_down1 = downconv(8, 8)
        # Set downconvolution layer 2
        self.conv_down2 = downconv(8, 16)
        # Set downconvolution layer 3
        self.conv_down3 = downconv(16, 24)
        # Set downconvolution layer 4
        self.conv_down4 = downconv(24, 32)
        # Set downconvolution layer 5
        self.conv_down5 = downconv(32, 40)
        # Set downconvolution layer 6
        self.conv_down6 = downconv(40, 48)

        # Set upconvolution layer 1
        self.conv_up1 = upconv(48, 320, res_ch=40)
        # Set upconvolution layer 2
        self.conv_up2 = upconv(320, 256, res_ch=32)
        # Set upconvolution layer 3
        self.conv_up3 = upconv(256, 192, res_ch=24)
        # Set upconvolution layer 4
        self.conv_up4 = upconv(192, 128, res_ch=16)
        # Set upconvolution layer 5
        self.conv_up5 = upconv(128, 64, res_ch=8)
        # Set upconvolution layer 6
        self.conv_up6 = upconv(64, 8, res_ch=8)

        # Set output layer
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

