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



#########################
correct_label = gt.unsqueeze_(0).view(1, height * width).type(torch.float32)
correct_label = correct_label  / correct_label.max()
correct_label.long()

reshaped_pred = pred[0,...]#.long()

# Count instances
unique_id = correct_label.reshape(correct_label.shape[1], correct_label.shape[0])
unique_labels = torch.unique(correct_label, sorted=True)
num_instances  = len(unique_labels)
counts = torch.histc(correct_label[0,:], bins=num_instances, min=0, max=1)
#counts = np.reshape(counts, (len(counts),1)).astype(float)

def unsorted_segment_sum(data, indexes):

    uid = torch.unique(indexes, sorted=True)
    cols = uid.shape[0]
    rows = data.shape[0]
    out = torch.zeros(rows, cols)

    for j in range(cols):

        aux = (indexes == uid[j]).float()
        out[:,j] = torch.sum(aux[:,0] * data, dim=1)

    return out

data = reshaped_pred.reshape(reshaped_pred.shape[1], reshaped_pred.shape[0])
index = unique_id.repeat(1,2)
index = index.reshape(index.shape[1], index.shape[0])

segmented_sum = unsorted_segment_sum(data, unique_id) # n_features, height x width
#segmented_sum_2 = torch.zeros(num_instances, 2).scatter_add(1, index.long(), data)

mu = torch.div(segmented_sum, counts)

#######################################
idx = torch.LongTensor(512*512,2).random_(0, 5)
src = torch.rand(5,2)

#idx = torch.LongTensor(2,3).random_(0, 3)
#src = torch.rand(100,3)

aux=torch.gather(src, 0, idx)

idx_ = torch.LongTensor(2,512*512).random_(0, 5)
data_ = torch.rand(2,512*512)
segmented_sum_2 = torch.zeros(2, 5).scatter_add(1, idx_, data_)


print(cluster_means)