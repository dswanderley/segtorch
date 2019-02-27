# -*- coding: utf-8 -*-
"""
Created on Wed Fev 27 18:37:04 2019

@author: Diego Wanderley
@python: 3.6
@description: Dataset loaders (images + ground truth)
"""   

import os
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image #,transform

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
        im_np = np.array(image).astype(np.float32) / 255.
        if (len(im_np.shape) > 2):
            im_np = im_np[:,:,0]

        # Grouth truth to array
        gt_np = np.array(gt_im).astype(np.float32)
        if (len(gt_np.shape) > 2): 
            gt_np = gt_np[:,:,0]

        # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
        gray_mask = (gt_np / 255.).astype(np.float32)
            
        # Multi mask - background (R = 1) / ovary (G = 1) / follicle (B = 1) 
        t1 = 128./2.
        t2 = 255. - t1
        multi_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3))
        # Background mask
        aux_b = multi_mask[:,:,0]
        aux_b[gt_np < t1] = 255.
        multi_mask[...,0] = aux_b
        # Ovary mask
        aux_o = multi_mask[:,:,1]
        aux_o[(gt_np >= t1) & (gt_np <= t2)] = 255.
        multi_mask[...,1] = aux_o
        # Follicle mask
        aux_f = multi_mask[:,:,2]
        aux_f[gt_np > t2] = 255.
        multi_mask[...,2] = aux_f
        # Convert to float
        multi_mask = (multi_mask / 255.).astype(np.float32)
                
        # Print data if necessary
        #Image.fromarray(gray_mask.astype(np.uint8)).save("gt.png")      
        #toprint = Image.fromarray(mask_rgb.astype(np.uint8))
        #toprint.save("multi_mask.png")

        # Apply transformations
        if self.transform:
            im_np, gray_mask, multi_mask = self.transform(im_np, gray_mask, multi_mask)
        
        # Convert to torch (to be used on DataLoader)
        return im_name, torch.from_numpy(im_np), torch.from_numpy(gray_mask), torch.from_numpy(multi_mask)

