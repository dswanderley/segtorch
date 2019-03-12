
import os
import time
from PIL import Image
from scipy import ndimage as ndi
import numpy as np

import torch
from torch.autograd import Variable


n_features = 6
delta_v = 0.5
delta_d = 1.5

# Load Image GT
path_gt ='cluster/gt.png'
img_gt = Image.open(path_gt)
img_gt.load()
# Parse image to numpy
data_gt = np.asarray(img_gt, dtype="float32") / 255.
data_gt = data_gt[...,1]
# Identify labels
inst_mask, _ = ndi.label(data_gt)
gt = torch.from_numpy(inst_mask)
height, width = gt.shape

# Adjust data
correct_label = gt.unsqueeze_(0).view(1, height * width)#.type(torch.float32)
correct_label.long()

# Prediction
pred = torch.rand(1, height*width, n_features)
reshaped_pred = pred.reshape(n_features, height*width)

# Count instances
unique_labels = torch.unique(correct_label, sorted=True) # instances labels (including background = 0)
num_instances  = len(unique_labels) # number of instances (including background)
counts = torch.histc(correct_label.float(), bins=num_instances, min=0, max=num_instances-1)
counts = counts.expand(n_features, num_instances)   # expected amount of pixel for each instance
unique_id = correct_label.expand(n_features, height * width).long() # expected index of each pixel

# Get sum by instance
segmented_sum = torch.zeros(n_features, num_instances).scatter_add(1, unique_id, reshaped_pred)
# Mean of each instance in each feature layer
mu = torch.div(segmented_sum, counts)
# Mean of the instance at each expected position for that instance
mu_expand = torch.gather(mu, 1, unique_id)


''' l_var  - intra-cluster distance '''

# Calculate intra distance
distance = mu_expand - reshaped_pred
distance = torch.norm(distance, dim=0) - delta_v    # apply delta_v
distance = torch.clamp(distance, 0., distance.max())**2 # max(0,x)
distance.reshape(1,len(distance))

l_var = torch.zeros(1, num_instances).scatter_add(1, unique_id[0].reshape(1, height*width), distance.reshape(1, height*width))
l_var = l_var / counts[0]
l_var = l_var.sum() / num_instances
print(l_var)


''' l_dist - inter-cluster distance'''

# Calculate inter distance
mu_sdim = mu.reshape(mu.shape[1]*mu.shape[0]) # reshape to apply meshgrid
mu_x, mu_y = torch.meshgrid(mu_sdim, mu_sdim)
aux_x = mu_x[:,:num_instances].reshape(n_features, num_instances, num_instances)#.permute(1,2,0)
aux_y = mu_y[:num_instances, :].reshape(num_instances, n_features, num_instances).permute(1,0,2)
# Calculate differece interclasses
mu_diff = aux_x - aux_y
mu_diff = torch.norm(mu_diff,dim=0)
# Use a matrix with delt_d to calculate each difference
aux_delta_d = 2 * delta_d * (torch.ones(mu_diff.shape) - torch.eye(mu_diff.shape[0])) # ignore diagonal (C_a = C_b)
l_dist = aux_delta_d - mu_diff
l_dist = torch.clamp(l_dist, 0., l_dist.max())**2 # max(0,x)
# sum / C(C-1)
l_dist = l_dist.sum() / num_instances / (num_instances - 1)

print(l_dist)
print('')