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




img_dir='Dataset/im/train/'
lbl_dir='Dataset/gt/train/'

image_paths = glob(os.path.join(img_dir, '*.png'))
label_paths = glob(os.path.join(lbl_dir, '*.png'))

image_paths.sort()
label_paths.sort()

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, label_paths, test_size=0.10, random_state=42)
print ('Number of train samples', len(y_train))
print ('Number of valid samples', len(y_valid))

bandwidth = 0.7


img = Image.open(y_valid[0])
img.load()
data = np.asarray( img, dtype="int32" )
valid_pred = data/255.
valid_pred[valid_pred < 1] = 0.
valid_pred = valid_pred[...,2]

correct_label, _ = ndi.label(valid_pred)


clt = cluster(valid_pred, bandwidth)

Image.fromarray((255*correct_label/4).astype(np.uint8)).save("instances.png")


correct_label = tf.reshape(correct_label, [512*512])
correct_label = tf.placeholder(dtype=tf.float32, shape=(None, 512, 512))

reshaped_pred = tf.reshape(correct_label, [512*512, 3])

unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
counts = tf.cast(counts, tf.float32)
num_instances = tf.size(unique_labels)


segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
mu_expand = tf.gather(mu, unique_id)


unique = np.unique(valid_pred)


instance_masks = get_instance_masks(valid_pred, bandwidth)

print('ok')