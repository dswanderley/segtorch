import os
import time
from PIL import Image
from glob import glob
from sklearn.cluster import MeanShift#, estimate_bandwidth
import numpy as np

import torch
from torch.autograd import Variable


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



def calculate_means(pred, gt, n_objects, max_n_objects):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # n_loc, n_objects, n_filters
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
        # n_loc, n_objects, 1
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            #if usegpu:
            #    _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        means.append(_mean_sample)

    means = torch.stack(means)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means


path_gt ='cluster/gt.png'
img_gt = Image.open(path_gt)
img_gt.load()

data_gt = np.asarray(img_gt, dtype="float32") / 255.
data_gt = data_gt[...,1]

inst_mask, num_inst = ndi.label(data_gt)
gt = torch.from_numpy(inst_mask)
#gt =   torch.rand(1,512*512,4)
height, width = gt.shape

tgt = np.zeros((height, width, num_inst))
for i in range(0, num_inst):
    aux = np.zeros((height, width))
    aux[inst_mask == i+1] = 1.
    tgt[...,i] = aux

target = torch.from_numpy(tgt.astype(np.float32))
target.unsqueeze_(1)
target = target.permute(0, 2, 3, 1).contiguous().view(1, height * width, num_inst)
# Pred
pred = torch.rand(1,512*512,2)
# Calc
cluster_means = calculate_means(pred, target, [4], 20)


print(cluster_means)

"""

#path_pred ='cluster/pred.png'
img_pred = Image.open(path_pred)
img_pred.load()

data_pred = np.asarray(img_pred, dtype="float32") / 255.
data_pred = data_pred[...,1]

h, w, feature_dim = data_pred.shape

num_clusters, labels, cluster_centers = cluster(data_pred.reshape([h*w, feature_dim]), bandwidth)

"""
