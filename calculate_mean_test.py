
import os
import time
from PIL import Image
from scipy import ndimage as ndi
import numpy as np

import torch
from torch.autograd import Variable


n_features = 2

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
reshaped_pred = pred.reshape(2, height*width)

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
mu_expander = torch.gather(mu, 1, unique_id)
