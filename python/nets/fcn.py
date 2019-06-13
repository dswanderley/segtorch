# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:51:31 2019

@author: Diego Wanderley
@python: 3.6
@description: Fully Convolutional Networks for Semantic Segmentation

"""

import torch
import torch.nn as nn
from torchvision import models

from modules import *


class FCN(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, softmax_out=True,
                        resnet_type=101, pretrained=False):
        super(FCN, self).__init__()

        self.resnet_type = resnet_type
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = None
        if n_channels != 3:
            self.inconv = FwdConv(n_channels, 3, kernel_size=1, padding=0)
        # Pre-trained model needs to be an identical network
        if pretrained:
            mid_classes = 21
        else:
            mid_classes = n_classes
        # Maind body
        if resnet_type == 50:    
            self.fcn_body = models.segmentation.fcn_resnet50(pretrained=False, num_classes=mid_classes)
            self.pretrained = False
        else:
            self.fcn_body = models.segmentation.fcn_resnet101(pretrained=pretrained, num_classes=mid_classes)

        if n_classes != 21:
            self.fcn_body.classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
            
            if  self.fcn_body.aux_classifier != None:
                self.fcn_body.aux_classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))

        # Softmax alternative
        self.has_softmax = softmax_out
        if softmax_out:
            self.softmax = nn.Softmax2d()
        else:
            self.softmax = None

    def forward(self, x):
        if self.inconv != None:
            x = self.inconv(x)
        x_deep = self.fcn_body(x)
        x_out = x_deep["out"]
        if self.softmax != None:
            x_out = self.softmax(x_out)
        return x_out



if __name__ == "__main__":
    model = FCN(n_channels=1, n_classes=3, resnet_type=101, pretrained=True)
    #model.eval()
    model.train()
    image = torch.randn(4, 1, 512, 512)

    output = model(image)
    print(output.size())
