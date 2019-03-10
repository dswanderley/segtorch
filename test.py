import os
import time
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import tensorflow as tf

from scipy import ndimage as ndi

# https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow

COLOR=[np.array([255,0,0]),
	   np.array([0,255,0]),
	   np.array([0,0,255]),
	   np.array([125,125,0]),
	   np.array([0,125,125]),
	   np.array([125,0,125]),
	   np.array([50,100,50]),
	   np.array([100,50,100])]


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


def get_instance_masks(prediction, bandwidth):
    batch_size = 1
    h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):
        num_clusters, labels, cluster_centers = cluster(prediction.reshape([h*w, feature_dim]), bandwidth)
        print ('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h,w])
        mask = np.zeros([h,w,3], dtype=np.uint8)

        num_clusters = min([num_clusters,8])
        for mask_id in range(num_clusters):
            ind = np.where(labels==mask_id)
            mask[ind] = COLOR[mask_id]

        instance_masks.append(mask)

    return instance_masks




def unsorted_segment_sum(data, indexes):

    uid = np.unique(indexes)
    out = np.zeros((uid.shape[0], data.shape[-1]))

    for j in range(data.shape[-1]):

        for i in range(uid.shape[0]):
            u = uid[i]

            idxs = np.zeros(indexes.shape)
            idxs[indexes == u] = 1

            sum_id = np.sum(idxs.T.dot(data))

            out[i][j] = sum_id

    return out


def gather_numpy(self, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


bandwidth = 0.7


# GT
path_gt ='cluster/gt.png'
img_gt = Image.open(path_gt)
img_gt.load()
data = np.asarray(img_gt, dtype="int32" )
gt = data/255.
gt[gt < 1] = 0.
gt = gt[...,2]

# Label
instance_labels, num_inst = ndi.label(gt)
instance_labels = instance_labels / num_inst

correct_label = np.reshape(instance_labels, (512*512,1))
# Pred
prediction = np.random.rand(512,512,2)
reshaped_pred = np.reshape(prediction, (512*512, 2))

# Count instances
unique_id = correct_label
unique_labels, counts = np.unique(correct_label, return_counts=True)
counts = np.reshape(counts, (len(counts),1)).astype(float)
num_instances  = len(unique_labels)

segmented_sum = unsorted_segment_sum(reshaped_pred, unique_id)

mu = np.divide(segmented_sum, counts)

mu_expand = gather_numpy(mu, 0, np.repeat(unique_id, mu.shape[1], axis = 1).astype(int))




aux=np.subtract(mu_expand, reshaped_pred)

