"""
high level support for doing this and that.
"""
#from __future__ import print_function, division
import os
#import torch
#import pandas as pd
from skimage import io
#,transform
#import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset#, DataLoader
#from torchvision import transforms, utils

class UltrasoundDataset(Dataset):
    """B-mode ultrasound dataset"""

    def __init__(self, im_dir='im', gt_dir='gt', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.images_name = os.listdir(self.im_dir)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):

        im_name = self.images_name[idx]

        im_path = os.path.join(self.im_dir, im_name)
        gt_path = os.path.join(self.gt_dir, im_name)

        image = io.imread(im_path)
        truth = io.imread(gt_path)

        sample = {'image': image, 'truth': truth}

        if self.transform:
            sample = self.transform(sample)

        return sample

OVARY_DATASET = UltrasoundDataset(im_dir='Dataset/im/', gt_dir='Dataset/gt/')

print(OVARY_DATASET)
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
#  shuffle=True, num_workers=threads, drop_last=True, pin_memory=True)
