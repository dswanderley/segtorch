# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 17:40:00 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for network prediction
"""

import os
import sys
import csv
import argparse
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nets.deeplab import DeepLabv3, DeepLabv3_plus
from nets.unet import *
from nets.gcn import *
from nets.fcn import *
from nets.rcnn import *
from utils.datasets import OvaryDataset
from utils.losses import DiceCoefficients


class Inference():
    """
        Inferecen class
    """

    def __init__(self, model, device, weights_path, batch_size=1,
                target=['gt_mask','ovary_mask'], folder='../predictions/'):
        '''
            Inference class - Constructor
        '''
        self.model = model
        self.device = device
        self.weights_path = weights_path
        self.batch_size = batch_size
        self._load_network()
        self.criterion = DiceCoefficients()
        if type(target) == list:
            self.target = target
        else:
            self.target = [target]
        self.pred_folder = folder + 'pred/'
        if not os.path.exists(self.pred_folder):
            os.makedirs(self.pred_folder)
        self.prob_folder = folder + 'prob/'
        if not os.path.exists(self.prob_folder):
            os.makedirs(self.prob_folder)


    def _load_network(self):
        '''
            Load weights and network state.
        '''

        if self.device.type == 'cpu':
            state = torch.load(self.weights_path, map_location='cpu')
        else:
            state = torch.load(self.weights_path)
        self.model.load_state_dict(state['state_dict'])
        self.state = state


    def _save_data(self, table_r):
        '''
            Save dice scores on a CSV file
        '''
        # Save results
        filename_r = self.pred_folder + "results.csv"
        with open(filename_r,'w') as fp:
            a = csv.writer(fp, delimiter=';')
            a.writerows(table_r)

        # Get training states
        filename_s = self.pred_folder + "states.csv"
        table_s = []
        if 'epoch' in self.state:
            table_s.append([
                'Best Val Epoch',
                self.state['epoch']
            ])
        if 'best_loss'in self.state:
            table_s.append([
                'Best Val Loss',
                self.state['best_loss']
            ])
        if 'arch'in self.state:
            table_s.append([
                'Architecture',
                self.state['arch']
            ])
        if 'n_input'in self.state:
            table_s.append([
                'Input channels',
                self.state['n_input']
            ])
        if 'target'in self.state:
            table_s.append([
                'Main task',
                self.state['target']
            ])
        if 'loss_function'in self.state:
            table_s.append([
                'Loss function',
                self.state['loss_function']
            ])
        if 'loss_weights'in self.state:
            table_s.append([
                'Loss function weights',
                self.state['loss_weights']
            ])
        if 'device'in self.state:
            table_s.append([
                'Device',
                self.state['device']
            ])

        # Save states
        with open(filename_s,'w') as fp:
            a = csv.writer(fp, delimiter=';')
            a.writerows(table_s)


    def predict(self, images, save=True):
        '''
            Predict segmentation function

            Arguments:
                @param images: Testset
                @param save: save images (True) - not implemented
        '''

        self.model.eval()
        self.model = self.model.to(self.device)

        dsc_data = []
        dsc_data.append(['name', 'backgound', 'stroma', 'follicles', 'ovary'])

        data_loader = DataLoader(images, batch_size=self.batch_size, shuffle=False)
        # Read images
        for _, sample in enumerate(data_loader):

            # Load data
            image = sample['image'].to(self.device)
            gt_mask = sample['gt_mask'].to(self.device)
            im_name = sample['im_name']
            # ovary prediction (interim)
            ov_mask = sample['ovary_mask'].to(self.device)  # load mask

            # data size
            bs, n_classes, height, width =  gt_mask.shape

            # Handle input
            if len(image.size()) < 4:
                image.unsqueeze_(1) # add a dimension to the tensor

            # Prediction
            pred = None
            with torch.no_grad():
                pred = self.model(image)



            # Handle multiples outputs
            if type(pred) is list:

                if type(pred[0]) is dict:
                    pred_dtct = [get_semantic_segmentation(pred, n_classes).to(self.device),
                                pred]
                    pred = pred_dtct
                # Main pred
                pred = pred[0]

            pred_max, pred_idx = pred.max(dim=1)
            pred_final = torch.clamp((pred - pred_max.unsqueeze_(1)) \
                                            + 0.0001, min=0)*10000

            # Compute Dice image by image
            for i in range(bs):

                # Get unique prediction
                pred_un = pred_final[i,...]
                pred_un.unsqueeze_(0)
                # Get unique GT
                gt_un = gt_mask[i,...]
                gt_un.unsqueeze_(0)
                ov_mask_un = ov_mask[i,...]
                ov_mask_un.unsqueeze_(0)

                # Evaluate - dice
                dsc = self.criterion(pred_un, gt_un)
                pred_ovary = torch.zeros(1, 2, height, width).to(self.device)
                pred_ovary[:,0,...] = pred_un[:,0,...]
                pred_ovary[:,1,...] = torch.clamp(pred_un[:,1,...] + pred_un[:,2,...],
                                                min=0, max=1)
                dsc_ov = self.criterion(pred_ovary, ov_mask_un)

                # Display evaluation
                iname = im_name[i]
                dsc_data.append([iname,
                                dsc[0].item(), dsc[1].item(), dsc[2].item(),
                                dsc_ov[1].item()])

                print('Filename:     {:s}'.format(iname))
                print('Stroma DSC:   {:f}'.format(dsc[1]))
                print('Follicle DSC: {:f}'.format(dsc[2]))
                print('Ovary DSC:    {:f}'.format(dsc_ov[1]))
                print('')
                # Save prediction
                img_out = pred_final[i].detach().cpu().permute(1,2,0).numpy()
                Image.fromarray((255*img_out).astype(np.uint8)).save( \
                                                self.pred_folder + iname)
                # Save probabilities
                img_prob = pred[i].detach().cpu().permute(1,2,0).numpy()
                Image.fromarray((255*img_prob).astype(np.uint8)).save( \
                                                    self.prob_folder + iname)

        self._save_data(dsc_data)



# Main calls
if __name__ == '__main__':

    # Load inputs
    parser = argparse.ArgumentParser(description="PyTorch segmentation network predictions \
        (only ovarian dataset).")
    parser.add_argument('--net', type=str, default='unet2',
                        choices=['fcn_r101', 'fcn_r50',
                                'deeplabv3', 'deeplabv3_r50', 'deeplabv3p', 'deeplabv3p_r50',
                                 'unet', 'unet_light', 'unet2', 'd_unet2',
                                 'sp_unet', 'sp_unet2',
                                 'gcn', 'gcn2', 'b_gcn', 'u_gcn',
                                 'mask_rcnn'],
                        help='network name (default: unet2)')
    parser.add_argument('--train_name', type=str, default='20190428_1133_unet2',
                        help='training name (default: 20190428_1133_unet2)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--folder_weigths', type=str, default='../weights/',
                        help='Weights root folder (default: ../weights/)')
    parser.add_argument('--folder_preds', type=str, default='../predictions/',
                        help='Predctions root folder (default: ../predictions/)')

    # Parse input data
    args = parser.parse_args()

    # Input parameters
    train_name = args.train_name
    net_type = args.net
    batch_size = args.batch_size
    folder_weights = args.folder_weigths
    folder_preds = args.folder_preds

    # Define input and output
    in_channels=1
    n_classes=3

    bilinear = False
    # Load Network model
    if net_type == 'mask_rcnn':
        n_classes = 3
        target = 'targets'
        loss = 'multitaskdict'
        train_with_targets = True
        model = MaskRCNN(n_channels=in_channels, n_classes=n_classes, pretrained=True)
    # FCN models
    elif net_type == 'fcn_r101':
        model = FCN(n_channels=in_channels, n_classes=n_classes, resnet_type=101)
    elif net_type == 'fcn_r50':
        model = FCN(n_channels=in_channels, n_classes=n_classes, resnet_type=50)
    # Deeplab v3
    elif net_type == 'deeplabv3':
        model = DeepLabv3(n_channels=in_channels, n_classes=n_classes, resnet_type=101)
    elif net_type == 'deeplabv3_r50':
        model = DeepLabv3(n_channels=in_channels, n_classes=n_classes, resnet_type=50)
    # Deeplab v3+
    elif net_type == 'deeplab_v3+':
        model = DeepLabv3_plus(nInputChannels=in_channels, n_classes=n_classes, os=16)
    elif net_type == 'deeplab_r50':
        model = DeepLabv3_plus(nInputChannels=in_channels, n_classes=n_classes, os=16, resnet_type=50)
    # Global Convolution Network
    elif net_type == 'gcn':
        model = GCN(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'gcn2':
        model = FCN_GCN(n_channels=in_channels, num_classes=n_classes)
    elif net_type == 'b_gcn':
        model = BalancedGCN(n_channels=in_channels, n_classes=n_classes)
    # Unet models
    elif net_type == 'unet':
        model = Unet(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'unet_light':
        model = UnetLight(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'sp_unet':
        model = SpatialPyramidUnet(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'sp_unet2':
        model = SpatialPyramidUnet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'd_unet':
        model = DilatedUnet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    else:
        model = Unet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset definitions
    dataset_test = OvaryDataset(im_dir='../datasets/ovarian/im/test/',
                                gt_dir='../datasets/ovarian/gt/test/')

    # Test network model
    print('Testing')
    print('')
    weights_path = folder_weights + train_name + '_weights.pth.tar'
    # Output folder
    out_folder = folder_preds + train_name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # Load inference
    inference = Inference(model, device, weights_path,
                    batch_size=batch_size, folder=out_folder)
    # Run inference
    inference.predict(dataset_test)