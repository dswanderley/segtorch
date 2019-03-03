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

from torchvision import transforms
from PIL import Image #,transform
from skimage import exposure, filters
from torch.utils.data import Dataset


class UltrasoundDataset(Dataset):
    """B-mode ultrasound dataset"""

    def __init__(self, im_dir='im', gt_dir='gt', one_hot=True, clahe=False, transform=None):
        """
        Args:
            im_dir (string): Directory with all the images.
            gt_dir (string): Directory with all the masks, with the same name of the original images.
            on_hot (bool): Optional output encoding one-hot-encoding or gray levels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.one_hot = one_hot
        self.clahe = clahe

        ldir_im = set(x for x in os.listdir(self.im_dir))
        ldir_gt = set(x for x in os.listdir(self.gt_dir))
        self.images_name = list(ldir_im.intersection(ldir_gt))


    def __len__(self):
        """
        @
        """
        return len(self.images_name)


    def __getitem__(self, idx):
        """
        @idx (int): file index.
        """

        '''
            Output encoding preparation
        '''
        # Output encod accepts two types: one-hot-encoding or gray scale levels
        # Variable encods contains a list of each data encoding: 1) Full GT mask, 2) Ovary mask, 3) Follicle mask
        if type(self.one_hot) is list:      # When a list is provided
            encods = []
            for i in range(3):
                if i > len(self.one_hot)-1: # If provided list is lower than expected
                    encods.append(True)
                else:
                    encods.append(self.one_hot[i])
        elif type(self.one_hot) is bool:    # When a single bool is provided
            encods = [self.one_hot, self.one_hot, self.one_hot]
        else:
            encods = [True, True, True]

        '''
            Load images
        '''
        # Image names: equal for original image and ground truth image
        im_name = self.images_name[idx]
        # Load Original Image (B-Mode)
        im_path = os.path.join(self.im_dir, im_name)    # PIL image in [0,255], 1 channel
        image = Image.open(im_path)
        # Load Ground Truth Image Image
        gt_path = os.path.join(self.gt_dir, im_name)    # PIL image in [0,255], 1 channel
        gt_im = Image.open(gt_path)

        # Apply transformations
        if self.transform:
            image, gt_im = self.transform(image, gt_im)

        '''
            Input Image preparation
        '''
        # Image to array
        im_np = np.array(image).astype(np.float32) / 255.
        if (len(im_np.shape) > 2):
            im_np = im_np[:,:,0]

        '''
            Main Ground Truth preparation - Gray scale GT and Multi-channels GT
        '''
        # Grouth truth to array
        gt_np = np.array(gt_im).astype(np.float32)
        if (len(gt_np.shape) > 2):
            gt_np = gt_np[:,:,0]


        # Multi mask - background (R = 255) / ovary (G = 255) / follicle (B = 255)
        t1 = 128./2.
        t2 = 255. - t1
        # Background mask
        mask_bkgound = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_bkgound[gt_np < t1] = 255.
        # Stroma mask
        mask_stroma = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_stroma[(gt_np >= t1) & (gt_np <= t2)] = 255.

        # Follicles mask
        mask_follicle = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_follicle[gt_np > t2] = 255.

        # Main mask output
        if encods[0]:
            # Multi mask - background (R = 1) / ovary (G = 1) / follicle (B = 1)
            multi_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3))
            multi_mask[...,0] = mask_bkgound
            multi_mask[...,1] = mask_stroma
            multi_mask[...,2] = mask_follicle
            gt_mask = (multi_mask / 255.).astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            gt_mask = (gt_np / 255.).astype(np.float32)

        '''
            Ovary Ground Truth preparation
        '''
        # Ovary mask
        mask_ovary = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_stroma[gt_np >= t1] = 255.

        # Ovarian auxiliary mask output
        if encods[1]:
            # Multi mask - background (R = 1) / ovary (G = 1)
            ov_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            ov_mask[...,0] = mask_bkgound
            ov_mask[...,1] = mask_stroma
            ov_mask = (ov_mask / 255.).astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            ov_mask = (mask_ovary / 255.).astype(np.float32)

        '''
            Follicles edge Ground Truth preparation
        '''
        mask_edges = mask_follicle
        # Ovarian auxiliary mask output
        if encods[2]:
            # Multi mask - background (R = 1) / follicle (G = 1)
            mask_fback = np.zeros((gt_np.shape[0], gt_np.shape[1]))
            mask_fback[gt_np < t2] = 255.
            # final mask
            fol_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            fol_mask[...,0] = mask_fback
            fol_mask[...,1] = mask_edges
            fol_mask = (fol_mask / 255.).astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            fol_mask = (mask_edges / 255.).astype(np.float32)

        '''
            Input data: Add CLAHE if necessary
        '''
        # Check has clahe
        if self.clahe:
            imclahe = np.zeros((im_np.shape[0], im_np.shape[1], 2))
            imclahe[...,0] = im_np
            imclahe[...,1] = exposure.equalize_adapthist(im_np, kernel_size=im_np.shape[0]/8, clip_limit=0.02, nbins=256)
            im_np = imclahe

        # Print data if necessary
        #Image.fromarray((255*im_np).astype(np.uint8)).save("im_np.png")
        #Image.fromarray((255*gt_mask).astype(np.uint8)).save("gt_all.png")
        #Image.fromarray((255*ov_mask[...,1]).astype(np.uint8)).save("gt_ov.png")
        #Image.fromarray((255*fol_mask[...,1]).astype(np.uint8)).save("gt_fol.png")

        # Convert to torch (to be used on DataLoader)
        return im_name, torch.from_numpy(im_np), torch.from_numpy(gt_mask), torch.from_numpy(ov_mask), torch.from_numpy(fol_mask)

