# -*- coding: utf-8 -*-
"""
Created on Wed Mar 03 17:40:00 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for network prediction
"""

import sys
import csv
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

            t = Variable(torch.Tensor([0.5])).to(self.device)
            pred_final = torch.where(pred < t, \
                    torch.zeros(pred.shape).to(self.device), \
                    torch.ones(pred.shape).to(self.device))
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

    # Model name
    train_name = '20190322_1721_Unet2'

    if(len(sys.argv)>1):
        train_name = sys.argv[1]
    print('train name:', train_name)

    # Load Unet
    model = Unet2(n_channels=1, n_classes=3)

    # Load CUDA if exist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset definitions
    dataset_test = OvaryDataset(im_dir='../dataset/im/test/', gt_dir='../dataset/gt/test/')

    # Test network model
    print('Testing')
    print('')
    weights_path = '../weights/' + train_name + '_weights.pth.tar'
    inference = Inference(model, device, weights_path)
    inference.predict(dataset_test)