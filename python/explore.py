# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:20:30 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script to analyze network weights and activations.
"""

import torch
import torchvision.models as models
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from nets.unet import *


folder_weights = '../weights/'
train_name = '20190428_1133_unet2'
weights_path = folder_weights + train_name + '_weights.pth.tar'

# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    state = torch.load(weights_path, map_location='cpu')
else:
    state = torch.load(weights_path)

# Random input
x = torch.rand(1,1,512,512)

def plot_kernels(weight):
    # Visualize conv filter
    kernels = weight.detach()
    fig, axarr = plt.subplots(kernels.size(0))
    for idx in range(kernels.size(0)):
        axarr[idx].imshow(kernels[idx].squeeze())
    plt.show()


#model = models.vgg11(pretrained=True)
model = Unet2(n_channels=1, n_classes=3, bilinear=False)
model.load_state_dict(state['state_dict'])

model.eval()
model = model.to(device)


filters = model.modules
body_model = [i for i in model.children()]
# layer1 = body_model[0][0]
layer1 = body_model[0].conv

#plot_kernels(layer1[0].weight)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model.conv_init.conv[0].register_forward_hook(get_activation('ext_conv1'))

output = model(x)

act = activation['ext_conv1'].squeeze()
num_plot = 4
fig, axarr = plt.subplots(min(act.size(0), num_plot))
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx])

plt.show()
# https://discuss.pytorch.org/t/visualize-feature-map/29597/2



from torchvision.utils import make_grid

kernels = model.conv_init.conv[0].weight.detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))
plt.show()


print('')


# https://stackoverflow.com/questions/52678215/find-input-that-maximises-output-of-a-neural-network-using-keras-and-tensorflow
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
