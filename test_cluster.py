import os
import time
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

import torch

from scipy import ndimage as ndi

# https://github.com/Wizaron/instance-segmentation-pytorch/blob/master/code/lib/losses/discriminative.py

def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth, bin_seeding=True)
    print ('Mean shift clustering, might take some time ...')
    tic = time.time()
    ms.fit(prediction)
    print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers



def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    pred_masked = pred_repeated * gt_expanded




eg0 = torch.rand(1,3,4,4)

eg1 = eg0.permute(0,2,3,1).contiguous().view(1,4*4,3)

eg2 = eg1.unsqueeze(2)

eg3 = eg1.unsqueeze(2).expand(1, 4*4, 3, 1)




path_gt ='cluster/gt.png'
path_pred ='cluster/pred.png'


img_gt = Image.open(path_gt)
img_pred = Image.open(path_pred)

img_gt.load()
img_pred.load()

data_gt = np.asarray(img_gt, dtype="float32") / 255.
data_gt = data_gt[...,1]
data_pred = np.asarray(img_pred, dtype="float32") / 255.
data_pred = data_pred[...,1]

h, w, feature_dim = data_pred.shape


num_clusters, labels, cluster_centers = cluster(data_pred.reshape([h*w, feature_dim]), bandwidth)