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
        self.inconv = fwdconv(n_channels, 3, kernel_size=1, padding=0)

        # Load Resnet
        resnet = models.resnet50(pretrained=True)

        # Set input layer
        self.conv0 = resnet.conv1 # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn0 = resnet.bn1 # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = resnet.relu
        self.maxpool = resnet.maxpool # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

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
        dc_x0 = self.conv0(c_x0)    # 512x512 -> 256x256
        dc_x0 = self.bn0(dc_x0)
        dc_x0 = self.relu0(dc_x0)
        # downstream
        dc_x1 = self.maxpool(dc_x0)
        dc_x1 = self.resnet1(dc_x1)     # 256x256 -> 128x128
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
        uc_x1   = self.deconv1(sc_x4)
        uc_x1_s = sc_x3 + uc_x1
        uc_x1_r = self.br5(uc_x1_s)
        #
        uc_x2   = self.deconv2(uc_x1_r)
        uc_x2_s = sc_x2 + uc_x2
        uc_x2_r = self.br6(uc_x2_s)
        #
        uc_x3   = self.deconv3(uc_x2_r)
        uc_x3_s = sc_x1 + uc_x3
        uc_x3_r = self.br7(uc_x3_s)
        #
        uc_x4   = self.deconv4(uc_x3_r)
        uc_x4_r = self.br8(uc_x4)
        #
        uc_x5   = self.deconv5(uc_x4_r)
        uc_x5_r = self.br9(uc_x5)

        # output
        x_out = self.softmax(uc_x5_r)

        return x_out


class BalancedGCN(nn.Module):
    '''
        Balanced Fully Global Convolution Network with simetric features reducing to number of classes.
    '''
    def __init__(self, n_channels, n_classes, bnorm=True, reg=True, convout=True):
        ''' Constructor '''
        super(BalancedGCN, self).__init__()
        # Number of classes definition
        self.n_classes = n_classes
        self.bnorm = bnorm
        self.reg = reg
        self.convout = convout

        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = fwdconv(n_channels, 3, kernel_size=1, padding=0)

        # Load Resnet
        resnet = models.resnet50(pretrained=True)

        # Set input layer
        self.conv0 = resnet.conv1 # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn0 = resnet.bn1 # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = resnet.relu
        self.maxpool = resnet.maxpool # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # Resnet layers
        self.resnet1 = resnet.layer1 # res-2 -> (depth) in:   64, out:  256
        self.resnet2 = resnet.layer2 # res-3 -> (depth) in:  256, out:  512
        self.resnet3 = resnet.layer3 # res-4 -> (depth) in:  512, out: 1024
        self.resnet4 = resnet.layer4 # res-5 -> (depth) in: 1024, out: 2048

        # GCN Layers
        self.gcn1 = globalconv(256,  64, k=55, batch_norm=bnorm, reg=reg, convout=convout)
        self.gcn2 = globalconv(512,  256, k=27, batch_norm=bnorm, reg=reg, convout=convout)
        self.gcn3 = globalconv(1024, 512, k=13, batch_norm=bnorm, reg=reg, convout=convout)
        self.gcn4 = globalconv(2048, 1024, k=7, batch_norm=bnorm, reg=reg, convout=convout)

        # Boundary Refine layers
        self.br1 = brconv(64, bnorm=bnorm, reg=reg, convout=convout)
        self.br2 = brconv(256, bnorm=bnorm, reg=reg, convout=convout)
        self.br3 = brconv(512, bnorm=bnorm, reg=reg, convout=convout)
        self.br4 = brconv(1024, bnorm=bnorm, reg=reg, convout=convout)
        self.br5 = brconv(1024, bnorm=bnorm, reg=reg, convout=convout)
        self.br6 = brconv(512, bnorm=bnorm, reg=reg, convout=convout)
        self.br7 = brconv(256, bnorm=bnorm, reg=reg, convout=convout)
        self.br8 = brconv(64, bnorm=bnorm, reg=reg, convout=convout)
        self.br9 = brconv(3, bnorm=bnorm, reg=reg, convout=convout)

        # Deconv
        self.deconv1 = upconv(2048, 1024) # Spatial size 16 x 16 -> 32 x 32
        self.deconv2 = upconv(1024, 512) # Spatial size 32 x 32 -> 64 x 64
        self.deconv3 = upconv(512, 256) # Spatial size 64 x 64 -> 128x128
        self.deconv4 = upconv(256, 64) # Spatial size 128x128 -> 256x256
        self.deconv5 = upconv(64, 3) # Spatial size 256x256 -> 512x512

        # Softmax
        self.softmax = nn.Softmax2d()


    def forward(self, x):
        ''' Foward method '''

        # input (adapt to resnet input 3ch)
        c_x0 = self.inconv(x)

        # Resnet
        dc_x0 = self.conv0(c_x0)    # 512x512 -> 256x256
        dc_x0 = self.bn0(dc_x0)
        dc_x0 = self.relu0(dc_x0)
        # downstream
        dc_x1 = self.maxpool(dc_x0)
        dc_x1 = self.resnet1(dc_x1)     # 256x256 -> 128x128
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
        uc_x1   = self.deconv1(sc_x4)
        uc_x1_s = sc_x3 + uc_x1
        uc_x1_r = self.br5(uc_x1_s)
        #
        uc_x2   = self.deconv2(uc_x1_r)
        uc_x2_s = sc_x2 + uc_x2
        uc_x2_r = self.br6(uc_x2_s)
        #
        uc_x3   = self.deconv3(uc_x2_r)
        uc_x3_s = sc_x1 + uc_x3
        uc_x3_r = self.br7(uc_x3_s)
        #
        uc_x4   = self.deconv4(uc_x3_r)
        uc_x4_r = self.br8(uc_x4)
        #
        uc_x5   = self.deconv5(uc_x4_r)
        uc_x5_r = self.br9(uc_x5)

        # output
        x_out = self.softmax(uc_x5_r)

        return x_out


class UGCN(nn.Module):
    '''
    U-net with Global convolutions class from end-to-end ovarian structures segmentation
    '''
    def __init__(self, n_channels, n_classes, bilinear=False):
        ''' Constructor '''
        super(UGCN, self).__init__()

        # Number of classes definition
        self.n_classes = n_classes

        # Set input layer
        self.conv_init  = inconv(n_channels, 8)

        # Set downconvolution layer 1
        self.conv_down1 = downconv(8, 8, dropout=0.2) # 512 -> 256
        # Set downconvolution layer 2
        self.conv_down2 = downconv(8, 16, dropout=0.2) # 256 - > 128
        # Set downconvolution layer 3
        self.conv_down3 = downconv(16, 24, dropout=0.2) # 128 -> 64
        # Set downconvolution layer 4
        self.conv_down4 = downconv(24, 32, dropout=0.2) # 64 -> 32
        # Set downconvolution layer 5
        self.conv_down5 = downconv(32, 40, dropout=0.2) # 32 -> 16
        # Set downconvolution layer 6
        self.conv_down6 = downconv(40, 48, dropout=0.2) # 16 -> 8

        # GCN Layers
        self.gcn0 = globalconv(8, 8, k=221, batch_norm=True, reg=True, convout=True) # 512
        self.gcn1 = globalconv(8, 8, k=111, batch_norm=True, reg=True, convout=True) # 256
        self.gcn2 = globalconv(16, 16, k=55, batch_norm=True, reg=True, convout=True) # 128
        self.gcn3 = globalconv(24, 24, k=27, batch_norm=True, reg=True, convout=True) # 64
        self.gcn4 = globalconv(32, 32, k=13, batch_norm=True, reg=True, convout=True) # 32
        self.gcn5 = globalconv(40, 40, k=7, batch_norm=True, reg=True, convout=True) # 16

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
        # Skip connections with GConv
        gc_x0 = self.gcn0(c_x0)
        gc_x1 = self.gcn1(dc_x1)
        gc_x2 = self.gcn2(dc_x2)
        gc_x3 = self.gcn3(dc_x3)
        gc_x4 = self.gcn4(dc_x4)
        gc_x5 = self.gcn5(dc_x5)
        # upstream
        uc_x1 = self.conv_up1(dc_x6, gc_x5)
        uc_x2 = self.conv_up2(uc_x1, gc_x4)
        uc_x3 = self.conv_up3(uc_x2, gc_x3)
        uc_x4 = self.conv_up4(uc_x3, gc_x2)
        uc_x5 = self.conv_up5(uc_x4, gc_x1)
        uc_x6 = self.conv_up6(uc_x5, gc_x0)
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


# Main calls
if __name__ == '__main__':

    net1 = GCN(1, 3)
    net2 = BalancedGCN(1, 3)
    net3 = UGCN(1, 3)
    print(net1)