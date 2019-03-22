# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 17:40:00 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for network prediction
"""

import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
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
        state = torch.load(self.weights_path)
        self.model.load_state_dict(state['state_dict'])


    def _save_data(self, table):
        '''
            Save dice scores on a CSV file
        '''
        filename = self.pred_folder + "results.csv"
        # Save table
        with open(filename,'w',newline='') as fp:
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
        dsc_data = []
        dsc_data.append(['name', 'backgound', 'stroma', 'follicles'])

        data_loader = DataLoader(images, batch_size=1, shuffle=False)
        for idx, sample in enumerate(data_loader):
            # Load data
            image = sample['image']
            gt_mask = sample['gt_mask']
            im_name = sample['im_name']
            # Active GPU
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
                image = image.to(self.device)
                gt_mask = gt_mask.to(self.device)
                #ov_mask = ov_mask.to(self.device)
                #fol_mask = fol_mask.to(self.device)

            # Handle with ground truth
            if len(gt_mask.size()) < 4:
                target = gt_mask.long()
            else:
                groundtruth = gt_mask.permute(0, 3, 1, 2).contiguous()

            # Prediction
            image.unsqueeze_(1) # add a dimension to the tensor
            pred = self.model(image)
            # Handle multiples outputs
            if type(pred) is list:
                pred = pred[0]

            dsc = self.criterion(pred, groundtruth)

            iname = im_name[0]
            dsc_data.append([iname, dsc[0].item(), dsc[1].item(), dsc[2].item()])

            print(iname)
            print('Stroma DSC:    {:f}'.format(dsc[1]))
            print('Follicle DSC:  {:f}'.format(dsc[2]))

            bs, cl, h, w = groundtruth.shape
            img_out = pred[0].detach().cpu().numpy()
            img_out = np.reshape(img_out, (h, w, cl))
            Image.fromarray((255*img_out).astype(np.uint8)).save(self.pred_folder + iname)

        self._save_data(dsc_data)