# -*- coding: utf-8 -*-
"""
Created on Wed Fev 08 00:40:00 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for network training
"""   

import os
import torch
import torchvision

import numpy as np
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from utils.datasets import UltrasoundDataset
from utils.losses import DiceLoss


'''
Transformation parameters
'''
#rotate_range = 25
#translate_range = (10.0, 10.0)
#scale_range = (0.90, 1.50)
#shear_range = 0.0
#im_size = (512,512)


def saveweights(state):
    '''
    Save network weights.

    Arguments:
    @state: parameters of the network
    '''
    path = ''
    filename = path + 'weights.pth.tar'
    
    torch.save(state, filename)


def train_net(net, epochs=30, batch_size=3, lr=0.1):
    '''
    Train network function

    Arguments:
        @param net: network model
        @param epochs: number of training epochs (int)
        @param batch_size: batch size (int)
        @param lr: learning rate
    '''

    # Load Dataset
    ovary_dataset = UltrasoundDataset(im_dir='Dataset/im/', gt_dir='Dataset/gt/')
    data_len = len(ovary_dataset)

    train_data = DataLoader(ovary_dataset, batch_size=batch_size, shuffle=True)
    # dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=threads, drop_last=True, pin_memory=True)
    
    # Define parameters
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) #optim.Adam(net.parameters())
    criterion =  DiceLoss() # nn.CrossEntropyLoss()
    best_loss = 1000    # Init best loss with a too high value

    # Run epochs
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # Active train
        net.train()
        # Init loss count
        loss_train_sum = 0
        
        for batch_idx, (im_name, image, gray_mask, multi_mask) in enumerate(train_data):
            
            # Active GPU train
            if torch.cuda.is_available():
                net = net.to(device)
                image = image.to(device)
                gray_mask = gray_mask.to(device)
                multi_mask = multi_mask.to(device)
            
            # Handle with ground truth
            if type(criterion) is type(nn.CrossEntropyLoss()):
                groundtruth = gray_mask.long()
            else:
                multi_mask = multi_mask.permute(0, 3, 1, 2).contiguous()
                groundtruth = multi_mask

            # Run prediction
            image.unsqueeze_(1) # add a dimension to the tensor, respecting the network input on the first postion (tensor[0])
            pred_masks = net(image)

            # Print output preview
            if batch_idx == len(train_data) - 1:
                torchvision.utils.save_image(image[0,...], "input.png")
                torchvision.utils.save_image(groundtruth[0,...], "groundtruth.png")
                torchvision.utils.save_image(pred_masks[0,...], "output.png")
            
            # Calculate loss for each batch
            loss = criterion(pred_masks, groundtruth)
            #loss = criterion(pred_masks[-1,...], groundtruth[-1,...])
            loss_train_sum += len(image) * loss.item()

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()           

        # Calculate average loss per epoch
        avg_loss_train = loss_train_sum / data_len
        print('loss: {:f}'.format(avg_loss_train))
        
        # To evaluate on validation set
        # XXXXXXXXXXXXXXXXXXXXX
        # call train()
        # epoch of training on the training set
        # call eval()
        # evaluate your model on the validation set
        # repeat
        # XXXXXXXXXXXXXXXXXXXXX

        # Save weights
        if best_loss > avg_loss_train:
            best_loss = avg_loss_train

            saveweights({
                        'epoch': epoch,
                        'arch': 'unet',
                        'state_dict': net.state_dict(),
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                        })



# if __name__ == '__main__':


# Load Unet
net = Unet(n_channels=1, n_classes=3)
print(net)

# Load CUDA if exist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_net(net)

