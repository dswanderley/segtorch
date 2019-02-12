"""
Train network
"""

import os
import torch

import numpy as np
import torch.nn as nn
#import matplotlib.pyplot as plt

from torch import optim
from PIL import Image #,transform

from torch.utils.data import Dataset, DataLoader

#from torchvision import transforms, utils
from unet import Unet

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
        """
        @
        """
        return len(self.images_name)

    def __getitem__(self, idx):
        """
        @
        """
        im_name = self.images_name[idx]

        im_path = os.path.join(self.im_dir, im_name)    # PIL image in [0,255], 3 channels
        gt_path = os.path.join(self.gt_dir, im_name)    # PIL image in [0,255], 3 channels
                
        image = Image.open(im_path)
        gt_im = Image.open(gt_path)

        # Image to array
        im_np = np.array(image)
        if (len(im_np.shape) > 2):
            im_np = im_np[:,:,0]

        # Three classes: background (0) / ovary  (127) / follicle (255)
        gt_np = np.array(gt_im)
        if (len(gt_np.shape) > 2):
            gt_np = gt_np[:,:,0]

        if self.transform:
            im_np, gt_np = self.transform(im_np, gt_np)
            
        return im_name, torch.from_numpy(im_np), torch.from_numpy(gt_np)


'''
Transformation parameters
'''
#rotate_range = 25
#translate_range = (10.0, 10.0)
#scale_range = (0.90, 1.50)
#shear_range = 0.0
#im_size = (512,512)

def train_net(net, epochs=1, batch_size=1, lr=0.1):

    # Load Dataset
    OVARY_DATASET = UltrasoundDataset(im_dir='Dataset/im/', gt_dir='Dataset/gt/')
    train_data = DataLoader(OVARY_DATASET, batch_size=4, shuffle=True)
    # dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=threads, drop_last=True, pin_memory=True)
    
    #optimizer = optim.Adam(net.parameters())

    #criterion = nn.BCELoss()

    # Run epochs
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        for batch_idx, (im_name, image, truth) in enumerate(train_data):

            #print(batch_idx)
            print(im_name)
            print(image.size())
            print(truth.size())

        #net.train()
        

               

        #for batch_idx in enumerate(train_data):
         #   print(batch_idx)
            #output = net(data)

        
        

# if __name__ == '__main__':


# Load Unet
net = Unet(n_channels=3, n_classes=2)

train_net(net)

