# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:20:30 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script to analyze network weights and activations.
"""

import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from nets.unet import *


folder_weights = '../weights/'
train_name = '20190503_1440_unet2'
weights_path = folder_weights + train_name + '_weights.pth.tar'

# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    state = torch.load(weights_path, map_location='cpu')
else:
    state = torch.load(weights_path)


def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    


#model = models.vgg11(pretrained=True)
model = Unet2(n_channels=1, n_classes=3, bilinear=False)

mm = model.double()
filters = mm.modules
body_model = [i for i in mm.children()][0]
#layer1 = body_model[0]
layer1 = body_model.conv[0]
tensor = layer1.weight.data.numpy()
plot_kernels(tensor)



# https://discuss.pytorch.org/t/visualize-feature-map/29597/2