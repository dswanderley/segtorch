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




bandwidth = 0.7


path_gt ='cluster/gt.png'
img_gt = Image.open(path_gt)
img_gt.load()
data = np.asarray(img_gt, dtype="int32" )
valid_pred = data/255.
valid_pred[valid_pred < 1] = 0.
valid_pred = valid_pred[...,2]

instance_labels, num_inst = ndi.label(valid_pred)
instance_labels = instance_labels / num_inst
Image.fromarray((255*instance_labels).astype(np.uint8)).save("instances.png")
#clt = cluster(valid_pred, bandwidth)

#correct_label = tf.placeholder(dtype=tf.float32, shape=(None, 512, 512))
#correct_label = tf.convert_to_tensor(instance_labels, dtype=tf.float32)


correct_label = np.reshape(instance_labels, (512*512,1))

unique_id = correct_label
unique_labels, counts = np.unique(correct_label, return_counts=True)
counts = counts.astype(float)
num_instances  = len(unique_labels)

#segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)


def unsorted_segment_sum(data, indexes):


    uid = np.unique(indexes)
    out = np.zeros(uid.shape)

    for i in range(uid.shape[0]):

        u = uid[i]

        idxs = np.zeros(indexes.shape)
        idxs[indexes == u] = 1
        
        sum_id = np.sum(idxs.T.dot(data))

        out[i] = sum_id

    return out


ooo = unsorted_segment_sum(correct_label, unique_id)


####### PRED #####
pred = np.random.rand(512, 512,2)
#pred_tf = tf.convert_to_tensor(pred, dtype=tf.float32)
#reshaped_pred = tf.reshape(pred_tf, [512*512, 2])

#### Unsorted_segmente_sum pytorch
'''
index = torch.tensor([[0, 0, 1, 1, 0, 1],
                      [1, 1, 0, 0, 1, 0]])
data = torch.tensor([[5., 1., 7., 2., 3., 4.],
                     [5., 1., 7., 2., 3., 4.]])

torch.zeros(2, 2).scatter_add(1, index, data)
> tensor([[  9.,  13.],
          [ 13.,   9.]])

'''








aux=np.array([1,2,3,4,5])
aux2=np.reshape(aux,(-1,1))

#### Loss ####
'''
unique_labels, unique_id, counts = tf.unique_with_counts(instance_labels)
counts = tf.cast(counts, tf.float32)
num_instances = tf.size(unique_labels)

print(num_instances)
'''