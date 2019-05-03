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
from nets.unet import Unet2
from utils.datasets import OvaryDataset
from utils.losses import DiceCoefficients


class Inference():
    """
        Inferecen class
    """

    def __init__(self, model, device, weights_path, folder='../predictions/'):
        '''
            Inference class - Constructor
        '''
        self.model = model
        self.device = device
        self.weights_path = weights_path
        self._load_network()
        self.criterion = DiceCoefficients()
        self.pred_folder = folder


    def _load_network(self):
        '''
            Load weights and network state.
        '''

        if self.device.type == 'cpu':
            state = torch.load(self.weights_path, map_location='cpu')
        else:
            state = torch.load(self.weights_path)
        self.model.load_state_dict(state['state_dict'])


    def _save_data(self, table):
        '''
            Save dice scores on a CSV file
        '''
        filename = self.pred_folder + "results.csv"
        # Save table
        with open(filename,'w') as fp:
            a = csv.writer(fp, delimiter=';')
            a.writerows(table)


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
        dsc_data.append(['name', 'backgound', 'stroma', 'follicles'])

        data_loader = DataLoader(images, batch_size=1, shuffle=False)
        for _, sample in enumerate(data_loader):
            # Load data
            image = sample['image'].to(self.device)
            gt_mask = sample['gt_mask'].to(self.device)
            im_name = sample['im_name']

            # Handle input
            if len(image.size()) < 4:
                image.unsqueeze_(1) # add a dimension to the tensor

            # Prediction
            pred = self.model(image)
            # Handle multiples outputs
            if type(pred) is list:
                pred = pred[0]

            pred_max, pred_idx = pred.max(dim=1)
            pred_final = torch.clamp((pred - pred_max.unsqueeze_(1)) + 0.0001, min=0)*10000

            # Evaluate - dice
            dsc = self.criterion(pred_final, gt_mask)

            # Display evaluation
            iname = im_name[0]
            dsc_data.append([iname, dsc[0].item(), dsc[1].item(), dsc[2].item()])

            print(iname)
            print('Stroma DSC:    {:f}'.format(dsc[1]))
            print('Follicle DSC:  {:f}'.format(dsc[2]))
            print('')
            # Save prediction
            img_out = pred_final[0].detach().cpu().permute(1,2,0).numpy()
            Image.fromarray((255*img_out).astype(np.uint8)).save(self.pred_folder + iname)

        self._save_data(dsc_data)



# Main calls
if __name__ == '__main__':

    # Load inputs
    parser = argparse.ArgumentParser(description="PyTorch segmentation network predictions (only ovarian dataset).")
    parser.add_argument('--net', type=str, default='unet2',
                        choices=['can', 'deeplab_v3+', 'unet', 'unet_light', 'unet2', 'd_unet2', 'gcn', 'gcn2', 'b_gcn', 'u_gcn'],
                        help='network name (default: unet2)')
    parser.add_argument('--train_name', type=str, default='20190428_1133_unet2',
                        help='training name (default: 20190428_1133_unet2)')
    parser.add_argument('--folder_weigths', type=str, default='../weights/',
                        help='Weights root folder (default: ../weights/)')
    parser.add_argument('--folder_preds', type=str, default='../predictions/',
                        help='Predctions root folder (default: ../predictions/)')

    # Parse input data
    args = parser.parse_args()

    # Input parameters
    train_name = args.train_name
    net_type = args.net
    folder_weights = args.folder_weigths
    folder_preds = args.folder_preds

    # Define input and output
    in_channels=1
    n_classes=3

    bilinear = False
     # Load Network model
    if net_type == 'can':
        model = CAN(in_channels, n_classes)
    elif net_type == 'deeplab_v3+':
        model = DeepLabv3_plus(nInputChannels=in_channels, n_classes=n_classes)
    elif net_type == 'gcn':
        model = GCN(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'gcn2':
        model = FCN_GCN(n_channels=in_channels, num_classes=n_classes)
    elif net_type == 'b_gcn':
        model = BalancedGCN(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'unet':
        model = Unet(n_channels=in_channels, n_classes=n_classes)
    elif net_type == 'unet_light':
            model = UnetLight(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    elif net_type == 'd_unet':
        model = DilatedUnet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
    else:
        model = Unet2(n_channels=in_channels, n_classes=n_classes, bilinear=bilinear)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset definitions
    dataset_test = OvaryDataset(im_dir='../datasets/ovarian/im/test/', gt_dir='../datasets/ovarian/gt/test/')

    # Test network model
    print('Testing')
    print('')
    weights_path = folder_weights + train_name + '_weights.pth.tar'
    # Output folder
    out_folder = folder_preds + train_name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # Load inference
    inference = Inference(model, device, weights_path, folder=out_folder)
    # Run inference
    inference.predict(dataset_test)