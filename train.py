"""
Train network
"""

import os
#import torch
from PIL import Image
#,transform
import numpy as np
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

        # Three classes: background (0) / ovary  (127) / follicle (255)
        gt_np = np.array(gt_im)

        truth = Image.fromarray(gt_indexes.astype('uint8'))        
        
        if self.transform:
            sample = self.transform(image, truth)

        return im_name, image, truth

'''
Transformation parameters
'''
#rotate_range = 25
#translate_range = (10.0, 10.0)
#scale_range = (0.90, 1.50)
#shear_range = 0.0
#im_size = (512,512)

def train_net(net, epochs=5, batch_size=1, lr=0.1):
    
    optimizer = optim.Adam(net.parameters())

    criterion = nn.BCELoss()

    train_data = DataLoader()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        

OVARY_DATASET = UltrasoundDataset(im_dir='Dataset/im/', gt_dir='Dataset/gt/')

print(OVARY_DATASET)
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
#  shuffle=True, num_workers=threads, drop_last=True, pin_memory=True)
