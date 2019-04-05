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
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(fwdconv, self).__init__()
        # Set conv layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
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


class globalconv(nn.Module):
    '''
        Global Convolutional module 
    '''
    def __init__(self, in_ch, m_ch, out_ch=None, k=7, batch_norm=False, reg=False, dropout=0):
        ''' Constructor '''
        super(globalconv, self).__init__()

        self.reg = reg
        self.batch_norm = batch_norm
        self.dropout = dropout
        if out_ch == None:
            out_ch = m_ch
        # Left side
        self.conv_left = nn.Sequential()
        # Conv 1
        self.conv_left.add_module("conv_l_1", nn.Conv2d(in_ch, m_ch, kernel_size=(k,1), padding=((k-1)/2,0)))
        if batch_norm:
            self.conv_left.add_module("bnorm_l_1",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_left.add_module("dropout_l_1", nn.Dropout2d(dropout))
        if reg:
            self.conv_left.add_module("relu_l_1",nn.ReLU(inplace=True))
        # Conv 2
        self.conv_left.add_module("conv_l_2", nn.Conv2d(m_ch, m_ch, kernel_size=(1,k), padding=(0,(k-1)/2))
        if batch_norm:
            self.conv_left.add_module("bnorm_l_2",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_left.add_module("dropout_l_2", nn.Dropout2d(dropout))
        if reg:
            self.conv_left.add_module("relu_l_2",nn.ReLU(inplace=True))

        # Right side
        self.conv_right = nn.Sequential()
        # Conv 1
        self.conv_right.add_module("conv_r_1", nn.Conv2d(in_ch, m_ch, kernel_size=(1,k), padding=(0,(k-1)/2))
        if batch_norm:
            self.conv_right.add_module("bnorm_r_1",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_right.add_module("dropout_r_1", nn.Dropout2d(dropout))
        self.conv_right.add_module("relu_r_1",nn.ReLU(inplace=True))
        # Conv 2
        self.conv_right.add_module("conv_r_2", nn.Conv2d(m_ch, m_ch, kernel_size=(k,1), padding=((k-1)/2,0))
        if batch_norm:
            self.conv_right.add_module("bnorm_r_2",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_right.add_module("dropout_r_2", nn.Dropout2d(dropout))
        if reg:
            self.conv_right.add_module("relu_r_2",nn.ReLU(inplace=True))

        # Conv sum
        self.conv_sum = nn.Sequential()
        self.conv_sum.add_module("conv_sum", nn.Conv2d(m_ch, out_ch, kernel_size=1, stride=1, padding=0))
        if batch_norm:
            self.conv_sum.add_module("bnorm_sum",nn.BatchNorm2d(out_ch))


    def forward(self, x):
        ''' Foward method '''
        x_l = self.conv_left(x)
        x_r = self.conv_right(x)
        # Sum
        x_gcb = x_l + x_r
        x_out = self.conv_sum(x_gcb)

        return x_out


class btneck_gconv(nn.Module):
    '''
        Global Convolutional bottleneck module 
    '''
    def __init__(self, in_ch, m_ch, k=7, batch_norm=True, reg=True, dropout=0):
        ''' Constructor '''
        super(btneck_gconv, self).__init__()
        self.reg = reg
        self.batch_norm = batch_norm
        self.dropout = dropout
        # Left side
        self.gconv = globalconv(in_ch, m_ch, out_ch=in_ch, k=k, batch_norm=batch_norm, reg=reg, dropout=dropout)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ''' Foward method '''
        x_gcb = self.gconv(x)
        # Sum
        x_out = x + x_gcb

        return x_out


class brconv(nn.Module):
    '''
        Boundary Refine Convolutional module 
    '''
    def __init__(self, out_ch, bnorm=False):
        ''' Constructor '''
        super(brconv, self).__init__()
        # Refined side
        self.conv_ref = nn.Sequential()
        # Conv 1 - 3x3 + Relu
        self.conv_ref.add_module("conv_1",  nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        if bnorm:
            self.conv_ref.add_module("bnorm_1", nn.BatchNorm2d(out_ch))
        self.conv_ref.add_module("relu", nn.ReLU(inplace=True))
        # Conv 2 - 3x3
        self.conv_ref.add_module("conv_2", nn.Conv2d(out_ch,out_ch, kernel_size=3, padding=1))
        if bnorm:
            self.conv_ref.add_module("bnorm_2", nn.BatchNorm2d(out_ch))
        
    def forward(self, x):
        ''' Foward method '''
        x_ref = self.conv_ref(x)
        # Sum
        x_out = x + x_ref
        
        return x_out
