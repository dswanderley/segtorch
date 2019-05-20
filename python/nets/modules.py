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


class InConv(nn.Module):
    '''
    Input layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(InConv, self).__init__()
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


class FwdConv(nn.Module):
    '''
    Foward convolution layer
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(FwdConv, self).__init__()
        # Set conv layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        if batch_norm:
            self.conv.add_module("bnorm_1", nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout_1", nn.Dropout2d(dropout))
        self.conv.add_module("relu_1", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    '''
    Downconvolution layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(DownConv, self).__init__()
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


class ConvPair(nn.Module):
    '''
    Enconding layer (conv + conv)
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(ConvPair, self).__init__()
        # Properties
        # Kernel size
        if type(kernel_size) == list or type(kernel_size) == tuple:
            self.kernel_size1 = kernel_size[0]
            self.kernel_size2 = kernel_size[1]
        else:
            self.kernel_size1 = kernel_size
            self.kernel_size2 = kernel_size
        # Zero Padding
        if type(padding) == list or type(padding) == tuple:
            self.padding1 = padding[0]
            self.padding2 = padding[1]
        else:
            self.padding1 = padding
            self.padding2 = padding
        # batch normalization
        if type(batch_norm) == list or type(batch_norm) == tuple:
            self.batch_norm1 = batch_norm[0]
            self.batch_norm2 = batch_norm[1]
        else:
            self.batch_norm1 = batch_norm
            self.batch_norm2 = batch_norm
        # dropout
        self.dropout = dropout
        # stride
        if type(stride) == list or type(stride) == tuple:
            self.stride1 = stride[0]
            self.stride2 = stride[1]
        else:
            self.stride1 = stride
            self.stride2 = stride
        # Init Sequential
        self.conv = nn.Sequential()
        # Conv layer 1
        self.conv.add_module("conv_1", nn.Conv2d(in_ch, out_ch, kernel_size, stride=self.stride1, padding=padding))
        if batch_norm:
            self.conv.add_module("bnorm_1", nn.BatchNorm2d(out_ch))
        self.conv.add_module("relu_1", nn.ReLU(inplace=True))
        # Conv layer 2
        self.conv.add_module("conv_2", nn.Conv2d(out_ch, out_ch, kernel_size, stride=self.stride2, padding=padding))
        if batch_norm:
            self.conv.add_module("bnorm_2", nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout", nn.Dropout2d(dropout))
        self.conv.add_module("relu_2", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class ConvSequence(nn.Module):
    '''
    Enconding layer (conv + conv)
    '''
    def __init__(self, in_ch, out_ch, n_vols, kernel_size=3, padding=1, stride=1, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(ConvSequence, self).__init__()
        # Properties
        self.num_volumes = n_vols
        # out_ch (output volumes)
        if type(out_ch) == list or type(out_ch) == tuple:
            self.depth = out_ch
        else:
            self.depth = [out_ch] * n_vols
        # in_ch (input volumes)
        self.input_volume = [in_ch] + self.depth[:-1]
        # Kernel size
        if type(kernel_size) == list or type(kernel_size) == tuple:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size] * n_vols
        # Zero Padding
        if type(padding) == list or type(padding) == tuple:
            self.padding = padding
        else:
            self.padding = [padding] * n_vols
        # batch normalization
        if type(batch_norm) == list or type(batch_norm) == tuple:
            self.batch_norm = batch_norm
        else:
            self.batch_norm = [batch_norm] * n_vols
        # dropout
        self.dropout = dropout
        # stride
        if type(stride) == list or type(stride) == tuple:
            self.stride = stride
        else:
            self.stride = [stride] * n_vols

        # Conv blocks
        self.conv = nn.Sequential()
        for i in range(n_vols):
            idx = str(i+1)
            ich = self.input_volume[i]
            och = self.depth[i]
            std = self.stride[i]
            bnorm = self.batch_norm[i]
            ksize = self.kernel_size[i]
            zpad = self.padding[i]
            # Add Convolution
            self.conv.add_module("conv_" + idx,
                                nn.Conv2d(ich, och, ksize, stride=std, padding=zpad)
                                )
            # Add Batch Normalization
            if bnorm:
                self.conv.add_module("bnorm_" + idx,
                                nn.BatchNorm2d(out_ch)
                                )
            # Add Dropout (to the last block)
            if (i == n_vols - 1) and (self.dropout > 0):
                self.conv.add_module("dropout",
                            nn.Dropout2d(dropout)
                            )
            # Add activation fucntion
            self.conv.add_module("relu_" + idx,
                            nn.ReLU(inplace=True)
                            )


    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    '''
    Upconvolution layer
    '''
    def __init__(self, in_ch, out_ch, res_ch=0, bilinear=False, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(UpConv, self).__init__()
        # Check interpolation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("fwdconv_1", FwdConv(in_ch+res_ch, out_ch, batch_norm=batch_norm, dropout=dropout))
        self.conv.add_module("fwdconv_2", FwdConv(out_ch, out_ch, batch_norm=batch_norm))

    def forward(self, x, x_res=None):
        ''' Foward method '''
        x_up = self.up(x)

        if x_res is None:
            x_cat = x_up
        else:
            x_cat = torch.cat((x_up, x_res), 1)

        x_conv = self.conv(x_cat)

        return x_conv


class OutConv(nn.Module):
    '''
    Output convolution layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(OutConv, self).__init__()
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


class GlobalAvgPool(nn.Module):
    '''
    Atrous or dilated convolution layer
    '''
    def __init__(self, in_ch, out_ch, batch_norm=True):
        ''' Constructor '''
        super(GlobalAvgPool, self).__init__()
        # parameters
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.batch_norm = batch_norm
        # Set sequential module
        self.conv = nn.Sequential()
        # pooling stencil size
        self.conv.add_module("avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        # 1x1 Convoltion (depth-weighted average)
        self.conv.add_module("conv", nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False))
        # Batch norm
        if batch_norm:
            self.conv.add_module("bnorm", nn.BatchNorm2d(out_ch))
        # relu
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        bs = x.shape[0] 
        if x.shape[0] == 1:
            x = torch.cat((x, x), dim=0)
        x = self.conv(x)

        if bs > 1:
            return x
        else:
            return x[0].unsqueeze_(0)


class AtrousConv(nn.Module):
    '''
    Atrous or dilated convolution layer
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=2, padding=1, batch_norm=True, dropout=0):
        ''' Constructor '''
        super(AtrousConv, self).__init__()
        # parameters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dilation = dilation
        self.padding = padding
        # Set conv layer
        self.conv = nn.Sequential()
        self.conv.add_module("conv", nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding, bias=False))
        if batch_norm:
            self.conv.add_module("bnorm",nn.BatchNorm2d(out_ch))
        if dropout > 0:
            self.conv.add_module("dropout", nn.Dropout2d(dropout))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x = self.conv(x)
        return x


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling
    '''
    def __init__(self, in_ch, out_ch, dilations):
        super(ASPP, self).__init__()
        # Properties
        self.dilations = dilations
        self.in_ch = in_ch
        self.out_ch = out_ch
        # List of atrous convolution
        self.atrous_list = []
        # Set each dilated convolution
        for i in range(len(dilations)):
            d = dilations[i]
            if d == 1:
                ks = 1
                pad = 0
            else:
                ks = 3
                pad = d
            # Sequential atrous convolution
            self.atrous_list.append(
                AtrousConv(in_ch, out_ch,
                            kernel_size=ks, dilation=d, padding=pad)
            )
        # Image Pooling
        self.im_pooling = GlobalAvgPool(in_ch, out_ch)

    def forward(self, x):
        # Set list to be concatenated
        y = []
        for i in range(len(self.dilations)):
            # Compute each dilated convolution
            conv = self.atrous_list[i]
            y.append(conv(x))
        # Image pooling
        x_pool = self.im_pooling(x)
        # Upsampling image average
        y.append(F.interpolate(x_pool, size=x.size()[2:], mode='bilinear', align_corners=True))
        # Concat volumes
        x_cat = torch.cat(y, dim=1)
        # Return concatenated volume
        return x_cat


class GlobalConv(nn.Module):
    '''
        Global Convolutional module
    '''
    def __init__(self, in_ch, m_ch, out_ch=None, k=7, batch_norm=False, reg=False, dropout=0, convout=False):
        ''' Constructor '''
        super(GlobalConv, self).__init__()

        self.reg = reg
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.convout = convout
        pad = int((k-1)/2)
        if out_ch == None:
            out_ch = m_ch
        # Left side
        self.conv_left = nn.Sequential()
        # Conv 1
        self.conv_left.add_module("conv_l_1", nn.Conv2d(in_ch, m_ch, kernel_size=(k,1), padding=(pad,0)))
        if batch_norm:
            self.conv_left.add_module("bnorm_l_1",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_left.add_module("dropout_l_1", nn.Dropout2d(dropout))
        if reg:
            self.conv_left.add_module("relu_l_1",nn.ReLU(inplace=True))
        # Conv 2
        self.conv_left.add_module("conv_l_2", nn.Conv2d(m_ch, m_ch, kernel_size=(1,k), padding=(0,pad)))
        if batch_norm:
            self.conv_left.add_module("bnorm_l_2",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_left.add_module("dropout_l_2", nn.Dropout2d(dropout))
        if reg:
            self.conv_left.add_module("relu_l_2",nn.ReLU(inplace=True))

        # Right side
        self.conv_right = nn.Sequential()
        # Conv 1
        self.conv_right.add_module("conv_r_1", nn.Conv2d(in_ch, m_ch, kernel_size=(1,k), padding=(0,pad)))
        if batch_norm:
            self.conv_right.add_module("bnorm_r_1",nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_right.add_module("dropout_r_1", nn.Dropout2d(dropout))
        self.conv_right.add_module("relu_r_1",nn.ReLU(inplace=True))
        # Conv 2
        self.conv_right.add_module("conv_r_2", nn.Conv2d(m_ch, m_ch, kernel_size=(k,1), padding=(pad,0)))
        if batch_norm:
            self.conv_right.add_module("bnorm_r_2", nn.BatchNorm2d(m_ch))
        if dropout > 0:
            self.conv_right.add_module("dropout_r_2", nn.Dropout2d(dropout))
        if reg:
            self.conv_right.add_module("relu_r_2",nn.ReLU(inplace=True))

        # Conv sum
        if self.convout:
            self.conv_sum = nn.Sequential()
            self.conv_sum.add_module("conv_sum", nn.Conv2d(m_ch, out_ch, kernel_size=1, stride=1, padding=0))
            if batch_norm:
                self.conv_sum.add_module("bnorm_sum", nn.BatchNorm2d(out_ch))
            if reg:
                self.conv_sum.add_module("relu_sum", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        x_l = self.conv_left(x)
        x_r = self.conv_right(x)
        # Sum
        x_out = x_l + x_r
        # Output
        if self.convout:
            x_out = self.conv_sum(x_out)

        return x_out


class Btneck_Gconv(nn.Module):
    '''
        Global Convolutional bottleneck module
    '''
    def __init__(self, in_ch, m_ch, k=7, batch_norm=True, reg=True, dropout=0):
        ''' Constructor '''
        super(Btneck_Gconv, self).__init__()
        self.reg = reg
        self.batch_norm = batch_norm
        self.dropout = dropout
        # Left side
        self.gconv = GlobalConv(in_ch, m_ch, out_ch=in_ch, k=k, batch_norm=batch_norm, reg=reg, dropout=dropout, outconv=True)
        if self.reg:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ''' Foward method '''
        x_gcb = self.gconv(x)
        # Sum
        x_out = x + x_gcb
        if self.reg:
            x_out = self.relu(x_out)
        return x_out


class BrConv(nn.Module):
    '''
        Boundary Refine Convolutional module
    '''
    def __init__(self, out_ch, bnorm=False, reg=False, convout=False):
        ''' Constructor '''
        super(BrConv, self).__init__()
        # Properties
        self.convout = convout
        self.reg = reg
        self.batch_norm = bnorm

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
        if reg:
            self.conv_ref.add_module("relu", nn.ReLU(inplace=True))

        # Conv sum, if needs for output (1x1)
        if self.convout:
            self.conv_sum = nn.Sequential()
            self.conv_sum.add_module("conv_sum", nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0))
            if bnorm:
                self.conv_sum.add_module("bnorm_sum", nn.BatchNorm2d(out_ch))
            if reg:
                self.conv_sum.add_module("relu_sum", nn.ReLU(inplace=True))

    def forward(self, x):
        ''' Foward method '''
        # Apply conv
        x_ref = self.conv_ref(x)
        # Sum
        x_out = x + x_ref
        if self.convout:
            x_out = self.conv_sum(x_out)

        return x_out



# Main calls
if __name__ == '__main__':

    dilations = [1, 6, 12, 18]
    aspp_block = ASPP(2048, 256, dilations=dilations)
    #print(block)
    x = torch.rand(2,2048,32,32)
    y = aspp_block(x)
    print(y)