# -*- coding: utf-8 -*-
"""
Created on Wed Fev 27 18:37:04 2019

@author: Diego Wanderley
@python: 3.6
@description: Dataset loaders (images + ground truth)

"""

import os
import torch
import random

import numpy as np

from PIL import Image
from skimage import exposure, filters
from torchvision import transforms
from torch.utils.data import Dataset

from scipy import ndimage as ndi


def select_clicks(fmap, rate=.7, margin=.5):
    '''
    Get one point for each follicle
    '''
    # Total of elements
    n_elements = fmap.max()
    # Convert to %
    margin_dist = int(margin * 100)

    points = []
    for j in range(1, n_elements+1):
        # Draw a value acording the initial probability rate
        goahead = np.random.choice(np.arange(2), p=[1-rate, rate])
        # Process oif follicle was selected
        if goahead > 0:
            aux_map = np.zeros(fmap.shape)
            aux_map[fmap==j] = 1
            # Compute Bouding Box
            slice_x, slice_y = ndi.find_objects(aux_map==1)[0]
            # Compute Center of max
            center = ndi.measurements.center_of_mass(aux_map)
            # Get bouding box height and width
            delta_x = slice_x.stop - slice_x.start
            delta_y = slice_y.stop - slice_y.start
            # Calculate
            margin = random.randint(-margin_dist, margin_dist) / 100.
            new_x = round(center[0] + margin * delta_x / 2)
            new_y = round(center[1] + margin * delta_y / 2)

            points.append((new_x, new_y))

    return points


def iteractive_map(points, height, width):
    '''
        Compute the Euclidean distance transformation of the provided points.
    '''
    # Reference data for meshgrid
    x = np.array(range(0, height))
    y = np.array(range(0, width))
    # One map for each point
    if len(points) > 0:
        dist_maps = np.ones((len(x), len(y), len(points)))
    else:
        dist_maps = np.ones((len(x), len(y), 1))
    # Compute maps
    for i in range(len(points)):
        p = points[i]
        xv, yv = np.meshgrid(np.power(p[1]-x, 2), np.power(p[0]-y, 2))
        dist_maps[...,i] =  np.clip(np.sqrt(xv + yv), 0, 255)
    # Get minimum value in the third axis
    psf_map = dist_maps.min(axis=2)

    return psf_map / 255.


class OvaryDataset(Dataset):
    """
    Dataset of ovarian structures from B-mode images.
    """

    def __init__(self, im_dir='im', gt_dir='gt',
    one_hot=True, clahe=False, imap=False, transform=None):
        """
        Args:
            im_dir (string): Directory with all the images.
            gt_dir (string): Directory with all the masks, with the same name of
            the original images.
            on_hot (bool): Optional output encoding one-hot-encoding or gray levels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.one_hot = one_hot
        self.clahe = clahe
        self.imap = imap

        ldir_im = set(x for x in os.listdir(self.im_dir))
        ldir_gt = set(x for x in os.listdir(self.gt_dir))
        self.images_name = list(ldir_im.intersection(ldir_gt))


    def __len__(self):
        """
            Get dataset length.
        """
        return len(self.images_name)


    def __getitem__(self, idx):
        """
            Get batch of images and related data.

            Args:
                @idx (int): file index.
            Returns:
                @sample (dict): im_name, image, gt_mask, ovary_mask,
                    follicle_mask, follicle_instances, num_follicles.
        """

        '''
            Output encoding preparation
        '''
        # Output encod accepts two types: one-hot-encoding or gray scale levels
        # Variable encods contains a list of each data encoding:
        # 1) Full GT mask, 2) Ovary mask, 3) Follicle mask
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
        mask_bkgound[gt_np < t1] = 1.
        # Stroma mask
        mask_stroma = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_stroma[(gt_np >= t1) & (gt_np <= t2)] = 1.

        # Follicles mask
        mask_follicle = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_follicle[gt_np > t2] = 1.

        # Main mask output
        if encods[0]:
            # Multi mask - background (R = 1) / ovary (G = 1) / follicle (B = 1)
            multi_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3))
            multi_mask[...,0] = mask_bkgound
            multi_mask[...,1] = mask_stroma
            multi_mask[...,2] = mask_follicle
            gt_mask = multi_mask.astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            gt_mask = gt_np.astype(np.float32)

        '''
            Ovary Ground Truth preparation
        '''
        # Ovary mask
        mask_ovary = np.zeros((gt_np.shape[0], gt_np.shape[1]))
        mask_ovary[gt_np >= t1] = 1.

        # Ovarian auxiliary mask output
        if encods[1]:
            # Multi mask - background (R = 1) / ovary (G = 1)
            ov_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            ov_mask[...,0] = mask_bkgound
            ov_mask[...,1] = mask_ovary
            ov_mask = ov_mask.astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            ov_mask = mask_ovary.astype(np.float32)

        '''
            Follicles edge Ground Truth preparation
        '''
        # 2*Dilate - 2*Erode
        f_erode = ndi.morphology.binary_erosion(mask_follicle)
        f_erode = ndi.morphology.binary_erosion(f_erode).astype(np.float32)
        f_dilate = ndi.morphology.binary_dilation(mask_follicle)
        f_dilate = ndi.morphology.binary_dilation(f_dilate).astype(np.float32)
        mask_edges = f_dilate - f_erode

        # Ovarian auxiliary mask output
        if encods[2]:
            # Multi mask - background (R = 1) / follicle (G = 1)
            # follicle mask
            fol_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            fol_mask[...,1] = mask_follicle
            fol_mask[...,0] = 1. - fol_mask[...,1]
            fol_mask = (fol_mask).astype(np.float32)
            # final edge
            fol_edge = np.zeros((gt_np.shape[0], gt_np.shape[1], 2))
            fol_edge[...,0] = 1. - mask_edges
            fol_edge[...,1] = mask_edges
            fol_edge = (fol_edge).astype(np.float32)
        else:
            # Gray mask - background (0/255) / ovary  (128/255) / follicle (255/255)
            fol_mask = mask_follicle.astype(np.float32)
            fol_edge = mask_edges.astype(np.float32)

        '''
            Instance Follicles mask
        '''
        # Get mask labeling each follicle from 1 to N value.
        inst_mask, num_inst = ndi.label(mask_follicle)

        '''
            Instance Bouding Boxes
        '''
        boxes = []
        labels = []
        # Ovary box
        slice_x, slice_y = ndi.find_objects(mask_ovary==1)[0]
        box = [slice_x.start, slice_y.start, slice_x.stop, slice_y.stop]
        boxes.append(box)
        labels.append(1)
        # Follicles boxes
        for i in range(1,num_inst+1):
            slice_x, slice_y = ndi.find_objects(inst_mask==i)[0]
            box = [slice_x.start, slice_y.start, slice_x.stop, slice_y.stop]
            boxes.append(box)
            labels.append(2)
        boxes = np.array(boxes)
        labels = np.array(labels)

        '''
            Interactive Object Selection
        '''
        # Check has interactive map
        if self.imap:
            if len(im_np.shape) == 2:
                im_np = im_np.reshape(im_np.shape+(1,))

            if type(self.imap) == list:
                selected_points = select_clicks(inst_mask, rate=self.imap[0], margin=self.imap[1])
            else:
                selected_points = select_clicks(inst_mask)
            imap_fol = iteractive_map(selected_points, im_np.shape[0], im_np.shape[1])
            imap_fol = imap_fol.reshape(imap_fol.shape+(1,))
            im_np = np.concatenate((im_np, imap_fol), axis=2).astype(np.float32)

        '''
            Input data: Add CLAHE if necessary
        '''
        # Check has clahe
        if self.clahe:
            if len(im_np.shape) == 2:
                im_np = im_np.reshape(im_np.shape+(1,))
            imclahe = np.zeros((im_np.shape[0], im_np.shape[1], 1))
            imclahe[...,0] = exposure.equalize_adapthist(im_np[...,0], kernel_size=im_np.shape[0]/8,
                            clip_limit=0.02, nbins=256)
            im_np = np.concatenate((imclahe, im_np), axis=2).astype(np.float32)

        # Print data if necessary
        #Image.fromarray((255*im_np).astype(np.uint8)).save("im_np.png")

        # Convert to torch (to be used on DataLoader)
        torch_im = torch.from_numpy(im_np)
        if len(torch_im.shape) > 2:
            torch_im = torch_im.permute(2, 0, 1).contiguous()

        torch_gt = torch.from_numpy(gt_mask)
        if len(torch_gt.shape) > 2:
            torch_gt = torch_gt.permute(2, 0, 1).contiguous()

        torch_ov = torch.from_numpy(ov_mask)
        if len(torch_ov.shape) > 2:
            torch_ov = torch_ov.permute(2, 0, 1).contiguous()

        torch_fol = torch.from_numpy(fol_mask)
        if len(torch_fol.shape) > 2:
            torch_fol = torch_fol.permute(2, 0, 1).contiguous()

        torch_edge = torch.from_numpy(fol_edge)
        if len(torch_edge.shape) > 2:
            torch_edge = torch_edge.permute(2, 0, 1).contiguous()

        torch_is = torch.from_numpy(inst_mask)
        if len(torch_is.shape) > 2:
            torch_is = torch_is.permute(2, 0, 1).contiguous()

        sample =  { 'im_name': im_name,
                    'image': torch_im,
                    'gt_mask': torch_gt,
                    'ovary_mask': torch_ov,
                    'follicle_mask': torch_fol,
                    'follicle_edge': torch_edge,
                    'follicle_instances': torch_is,
                    'num_follicles':  num_inst,
                    'boxes': boxes,
                    'labels': labels
                     }

        return sample


class VOC2012Dataset(Dataset):
    """
    Dataset of Pascal VCO 2012 images.
    """

    def __init__(self, im_dir, gt_dir, file_list, one_hot=True, transform=None):
        """
        Args:
            im_dir (string): Directory with all the images.
            gt_dir (string): Directory with all the masks, with the same name of
            the original images.
            on_hot (bool): Optional output encoding one-hot-encoding or gray levels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.one_hot = one_hot
        self.n_classes = 20+2
        self.height = 512
        self.width = 512
        self.colormap = [[ 0,   0,   0], # backgroud
                        [128,   0,   0], # aeroplane
                        [  0, 128,   0], # bicycle
                        [128, 128,   0], # bird
                        [  0,   0, 128], # boat
                        [128,   0, 128], # bottle
                        [  0, 128, 128], # bus
                        [128, 128, 128], # car
                        [ 64,   0,   0], # cat
                        [192,   0,   0], # chair
                        [ 64, 128,   0], # cow
                        [192, 128,   0], # diningtable
                        [ 64,   0, 128], # dog
                        [192,   0, 128], # horse
                        [ 64, 128, 128], # motorbike
                        [192, 128, 128], # person
                        [  0,  64,   0], # potterdplant
                        [128,  64,   0], # sheep
                        [  0, 192,   0], # sofa
                        [128, 192,   0], # train
                        [  0,  64, 128], # tv monitor
                        [224, 224, 192], # void
                        ]

        with open(file_list) as f:
            flist = f.read().splitlines()
        self.images_name = flist


    def __len__(self):
        """
            Get dataset length.
        """
        return len(self.images_name)


    def __getitem__(self, idx):
        """
            Get batch of images and related data.

            Args:
                @idx (int): file index.
            Returns:
                @sample (dict): im_name, image, gt_mask, ovary_mask,
                    follicle_mask, follicle_instances, num_follicles.
        """

        '''
            Output encoding preparation
        '''
        # Output encod accepts two types: one-hot-encoding or gray scale levels

        '''
            Load images
        '''
        # Image names: equal for original image and ground truth image
        im_name = self.images_name[idx]
        jpg_name = im_name + '.jpg'
        # Load Original Image (RGB)
        im_path = os.path.join(self.im_dir, jpg_name)    # PIL image in [0,255], 1 channel
        image = Image.open(im_path)
        # Ground Truth Image
        gt_name = im_name + '.png'
        # Load Ground Truth  Image (RGB)
        gt_path = os.path.join(self.gt_dir, gt_name)    # PIL image in [0,255], 1 channel
        gt_im = Image.open(gt_path)

        # Apply transformations
        if self.transform:
            image, gt_im = self.transform(image, gt_im)

        '''
            Input Image preparation
        '''
        # Image to array
        im_np = np.array(image).astype(np.float32) / 255.
        h, w, d = im_np.shape
        im_square = np.zeros((self.height, self.width, d))

        p_left = int(round((self.width - w) / 2))
        p_right = int(p_left + w)
        p_top = int(round((self.height - h) / 2))
        p_down = int(p_top + h)

        im_box = {'top': p_top,
                'bottom': p_down,
                'left': p_left,
                'right': p_right}

        im_square[p_top:p_down, p_left:p_right, :] = im_np


        '''
            Main Ground Truth preparation - Gray scale GT and Multi-channels GT
        '''
        # Grouth truth to array
        gt_np = np.array(gt_im).astype(np.float32)
        # Get class with more incidences
        hist = gt_im.histogram()
        main_class = np.argmax(hist[1:-1]) + 1 # ignore void and background
        # Apply gt to square
        gt_square_p = np.zeros((self.height, self.width))
        gt_square_p[p_top:p_down, p_left:p_right] = gt_np
        # Define lables from 0 (background) to n_classes-1 and add void idx
        labels_idx = list(range(self.n_classes-1)) +  [255]
        if self.one_hot:
            gt_square = np.zeros((self.height, self.width, self.n_classes))
            for c in range(self.n_classes):
                mask = np.zeros((gt_square.shape[0], gt_square.shape[1]))
                mask[(gt_square_p == labels_idx[c])] = 1.
                gt_square[...,c] = mask
        else:
            gt_square = gt_square_p
            gt_square[gt_square_p == 255] = self.n_classes-1

        # Print data if necessary
        #Image.fromarray((255*im_np).astype(np.uint8)).save("im_np.png")
        #Image.fromarray((255*gt_mask).astype(np.uint8)).save("gt_all.png")

        torch_im = torch.from_numpy(im_square.astype(np.float32))
        if len(torch_im.shape) > 2:
            torch_im = torch_im.permute(2, 0, 1).contiguous()

        torch_gt = torch.from_numpy(gt_square.astype(np.float32))
        if len(torch_gt.shape) > 2:
            torch_gt = torch_gt.permute(2, 0, 1).contiguous()


        sample =  { 'im_name': im_name,
                    'image': torch_im,
                    'gt_mask': torch_gt,
                    'main_class': main_class,
                    'im_box':  im_box }

        return sample



# Main calls
if __name__ == '__main__':

    from torch.utils.data import DataLoader

    # pre-set
    dataset = OvaryDataset(im_dir='../datasets/ovarian/im/test/',
                           gt_dir='../datasets/ovarian/gt/test/',
                           imap=False, clahe=False, transform=False)
    # Loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # iterate
    for _, sample in enumerate(data_loader):
            # Load data
            images = sample['image']
            boxes = sample['boxes']
            labels = sample['labels']

            targets = {
                'boxes': boxes,
                'labels': labels
            }

            images = list(image for image in images)

            print('')

