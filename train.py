import os

import numpy as np
import torch.nn as nn

from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from unet import Unet

folder_root = 'E:/GynUS/Simluations/Dataset/'
folder_im = 'Images/'
folder_gt = 'Masks/'
folder_train = 'Train/'
folder_val = 'Val/'
# im_folder_test = 'Test/'


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def load_dataset(dataset):

    # Get folders
    im_dir = folder_root + folder_im + dataset
    gt_dir = folder_root + folder_gt + dataset
    # List data
    im_list = set(x for x in os.listdir(im_dir))
    gt_list = set(y for y in os.listdir(gt_dir))
    # Compare data
    data_list = list(im_list & gt_list)


    for fname in data_list:
        im = Image.open(im_dir + fname)
        gt = Image.open(gt_dir + fname)

        yield zip(im, gt)


def train_net(net, epochs=5, batch_size=1, lr=0.1):
    
    optimizer = optim.Adam(net.parameters())

    criterion = nn.BCELoss()

    train_data = DataLoader()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()

        #train_data = load_dataset(folder_train)
        
        #for im, bt in enumerate(batch(train_data, batch_size)):
         #   print(im)



if __name__ == '__main__':


    # Load Unet
    net = Unet(n_channels=3, n_classes=2)
    # print(net)
    
    train_net(net)
    