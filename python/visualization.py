
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:24:10 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for visualize trainned networks
"""

import os
import torch
from PIL import Image
from explore.cnn_layer_visualization import *
from explore.guided_backprop import *
from nets.unet import *
from nets.deeplab import *

####################################

# input parameters
cnn_layer = 1
filter_pos = 3
net_type = 'unet2'

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





example_index=3
# Pick one of the examples
example_list = (('../datasets/examples/snake.jpg', 56),
                ('../datasets/examples/cat_dog.png', 243),
                ('../datasets/examples/spider.png', 72),
                ('../datasets/examples/ultrasound.png', 100))
img_path = example_list[example_index][0]
target_class = example_list[example_index][1]
file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
# Read image
original_image = Image.open(img_path).convert('L')
numpy_image = np.array(original_image).astype(np.float32) 
# Process image
prep_img = preprocess_image(np.expand_dims(numpy_image, axis=2), resize_im=False)
# Define model
pretrained_model = new_model




 # Guided backprop
GBP = GuidedBackprop(pretrained_model)
# Get gradients
guided_grads = GBP.generate_gradients(prep_img, target_class)
# Save colored gradients
save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
# Convert to grayscale
grayscale_guided_grads = convert_to_grayscale(guided_grads)
# Save grayscale gradients
save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
# Positive and negative saliency maps
pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
print('Guided backprop completed')











# Load visualization class
layer_vis = CNNLayerVisualization(new_model, cnn_layer, filter_pos)
# Layer visualization with pytorch hooks
layer_vis.visualise_layer_with_hooks()