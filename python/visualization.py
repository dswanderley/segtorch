
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:24:10 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for visualize trainned networks
"""

import os
import torch
from explore.cnn_layer_visualization import *
from nets.unet import *
from nets.deeplab import *

####################################

# input parameters
cnn_layer = 1
filter_pos = 3
net_type = 'deeplab_v3+'

folder_weights = '../weights/'

# Set network
if net_type == 'unet2':
    model = Unet2(n_channels=1, n_classes=3, bilinear=False)
    train_name = '20190428_1133_unet2'
else:
    model = DeepLabv3_plus(nInputChannels=1, n_classes=3, os=16, pretrained=False)
    train_name = '20190506_1145_deeplab_v3+'

# Weights file
weights_path = folder_weights + train_name + '_weights.pth.tar'

# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set device
if device.type == 'cpu':
    state = torch.load(weights_path, map_location='cpu')
else:
    state = torch.load(weights_path)

# Load weights
model.load_state_dict(state['state_dict'])
model = model.to(device)

# Change model to a list type
new_model = nn.Sequential(*list(model.children()))

# Load visualization class
layer_vis = CNNLayerVisualization(new_model, cnn_layer, filter_pos)
# Layer visualization with pytorch hooks
layer_vis.visualise_layer_with_hooks()