# -*- coding: utf-8 -*-
"""
Created on Fri Jun 7 16:38:30 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script to analyze network weights and activations.
"""

import torch
import numpy as np

import torchvision.models as models

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy import misc, ndimage
from torch.autograd import Variable

from nets.unet import *


# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        if device.type == 'cpu':
            self.features = output.clone().detach().requires_grad_(True)
            #torch.tensor(output, requires_grad=True)
        else:
            self.features = output.clone().detach().requires_grad_(True).cuda()
            #torch.tensor(output, requires_grad=True).cuda()
    def close(self):
        self.hook.remove()


class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = models.vgg16(pretrained=True).to(device).eval()
        
    def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
        sz = self.size
        
        img = torch.rand(1,3,sz,sz)
        activations = SaveFeatures(list(self.model.children())[0][layer])  # register hook

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times

            img_var = Variable(img)  # convert image to Variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(20):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            img_np = img_var.data.cpu().numpy()[0].transpose(1,2,0)
            self.output = img
            sz = int(self.upscaling_factor * sz)+1  # calculate new image size
            img_np = misc.imresize(img_np, (sz, sz), interp='bicubic')  # scale image up
            if blur is not None: img_np = ndimage.uniform_filter(img_np, size=blur)  # blur image to reduce high frequency patterns
            img = torch.from_numpy(img_np).permute(2,0,1).unsqueeze_(0)

            
            plt.imshow(img_np)
            plt.show()


        self.save(layer, filter)
        activations.close()
        
    def save(self, layer, filter):
        plt.imsave("layer_"+str(layer)+"_filter_"+str(filter)+".png", np.clip(self.output, 0, 1))


layer = 30-1
filter = 180

FV = FilterVisualizer(size=56, upscaling_steps=12, upscaling_factor=1.2)
FV.visualize(layer, filter)

img = PIL.Image.open("layer_"+str(layer)+"_filter_"+str(filter)+".jpg")
plt.figure(figsize=(7,7))
plt.grid(None)
plt.imshow(img)