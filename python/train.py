# -*- coding: utf-8 -*-
"""
Created on Wed Fev 08 00:40:00 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for network training
"""

import torch
from torch.utils.data import DataLoader

class Training:
    """
        Training classe
    """

    def __init__(self, model, device, train_set, valid_set,
                opt, loss, logger=None, train_name='net'):
        '''
            Training class - Constructor
        '''
        self.model = model
        self.device = device
        self.dataset_train = train_set
        self.dataset_val = valid_set
        self.optimizer = opt
        self.criterion = loss
        self.logger = logger
        self.train_name = train_name

    def _saveweights(self, state):
        '''
            Save network weights.

            Arguments:
            @state (dict): parameters of the network
        '''
        path = '../weights/'
        filename = path + self.train_name + '_weights.pth.tar'
        torch.save(state, filename)


    def _iterate_train(self, data_loader_train):

        # Init loss count
        loss_train_sum = 0
        data_train_len = len(self.dataset_train)

        # Active train
        self.model.train()

        # Batch iteration - Training dataset
        #for batch_idx, (im_name, image, gt_mask, ov_mask, fol_mask) in enumerate(data_loader_train):
        for batch_idx, sample in enumerate(data_loader_train):
            # Load data
            image = sample['image']
            gt_mask = sample['gt_mask']
            #gt_mask = sample['follicle_instances']
            # Active GPU train
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
                image = image.to(self.device)
                gt_mask = gt_mask.to(self.device)
                #ov_mask = ov_mask.to(self.device)
                #fol_mask = fol_mask.to(self.device)

            # Handle with ground truth
            if len(gt_mask.size()) < 4:
                groundtruth = gt_mask.long()
            else:
                groundtruth = gt_mask.permute(0, 3, 1, 2).contiguous()

            # Run prediction
            image.unsqueeze_(1) # add a dimension to the tensor
            pred_masks = self.model(image)
            # Handle multiples outputs
            if type(pred_masks) is list:
                pred_masks = pred_masks[0]
                #pred_masks = pred_masks[1]

            # Output preview
            if batch_idx == len(data_loader_train) - 1:
                ref_image_train = image[0,...]
                ref_pred_train = pred_masks[0,...]

            # Calculate loss for each batch
            loss = self.criterion(pred_masks, groundtruth)
            loss_train_sum += len(image) * loss.item()

            # Update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Calculate average loss per epoch
        avg_loss_train = loss_train_sum / data_train_len

        return avg_loss_train, ref_image_train, ref_pred_train


    def _iterate_val(self, data_loader_val):

        # Init loss count
        loss_val_sum = 0
        data_val_len = len(self.dataset_val)

        # To evaluate on validation set
        self.model.eval()

        # Batch iteration - Validation dataset
        for batch_idx, sample in enumerate(data_loader_val):
            # Load data
            image = sample['image']
            gt_mask = sample['gt_mask']
            #gt_mask = sample['follicle_instances']
            # Active GPU
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
                image = image.to(self.device)
                gt_mask = gt_mask.to(self.device)
                #ov_mask = ov_mask.to(self.device)
                #fol_mask = fol_mask.to(self.device)

            # Handle with ground truth
            if len(gt_mask.size()) < 4:
                groundtruth = gt_mask.long()
            else:
                groundtruth = gt_mask.permute(0, 3, 1, 2).contiguous()

            # Prediction
            self.optimizer.zero_grad()
            image.unsqueeze_(1) # add a dimension to the tensor, respecting the network input on the first postion (tensor[0])
            val_masks = self.model(image)
            # Handle multiples outputs
            if type(val_masks) is list:
                val_masks = val_masks[0]

            # Print output preview
            if batch_idx == len(data_loader_val) - 1:
                ref_image_val = image[0,...]
                ref_pred_val = val_masks[0,...]

            # Calculate loss for each batch
            val_loss = self.criterion(val_masks, groundtruth)
            loss_val_sum += len(image) * val_loss.item()

        # Calculate average validation loss per epoch
        avg_loss_val = loss_val_sum / data_val_len

        return avg_loss_val, ref_image_val, ref_pred_val


    def _logging(self, epoch, avg_loss_train, avg_loss_val,
                ref_image_train, ref_pred_train, ref_image_val, ref_pred_val):

        # 1. Log scalar values (scalar summary)
            info = { 'avg_loss_train': avg_loss_train,
                    'avg_loss_valid': avg_loss_val
                }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, epoch+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                if not value.grad is None:
                    self.logger.histo_summary(tag +'/grad', value.grad.data.cpu().numpy(), epoch+1)

            # 3. Log training images (image summary)
            info = {'train_image': ref_image_train.cpu().numpy(),
                    'train_predi': ref_pred_train.cpu().detach().numpy(),
                    'valid_image': ref_image_val.cpu().numpy(),
                    'valid_predi': ref_pred_val.cpu().detach().numpy()
                }
            for tag, im in info.items():
                self.logger.image_summary(tag, im, epoch+1)


    def train(self, epochs=100, batch_size=4):
        '''
        Train network function

        Arguments:
            @param net: network model
            @param epochs: number of training epochs (int)
            @param batch_size: batch size (int)
        '''

        # Load Dataset
        data_loader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True)
        data_loader_val = DataLoader(self.dataset_val, batch_size=1, shuffle=False)

        # Define parameters
        best_loss = 1000    # Init best loss with a too high value

        # Run epochs
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

            # ========================= Training =============================== #
            avg_loss_train, ref_image_train, ref_pred_train = self._iterate_train(data_loader_train)
            print('training loss:  {:f}'.format(avg_loss_train))

            # ========================= Validation ============================= #
            avg_loss_val, ref_image_val, ref_pred_val = self._iterate_val(data_loader_val)
            print('validation loss: {:f}'.format(avg_loss_val))

            # ======================== Save weights ============================ #
            if best_loss > avg_loss_val:
                best_loss = avg_loss_val
                # save
                self._saveweights({
                            'epoch': epoch,
                            'arch': 'unet',
                            'state_dict': self.model.state_dict(),
                            'best_loss': best_loss,
                            'optimizer': self.optimizer.state_dict()
                            })

            # ====================== Tensorboard Logging ======================= #
            if self.logger:
                self._logging(epoch, avg_loss_train, avg_loss_val,
                    ref_image_train, ref_pred_train, ref_image_val, ref_pred_val)