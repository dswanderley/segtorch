# -*- coding: utf-8 -*-
"""
Created on Wed Fev 08 00:40:00 2019

@author: Diego Wanderley
@python: 3.6 and Pytroch
@description: Script for network training
"""

import torch
from torch.utils.data import DataLoader
from utils.datasets import collate_fn_ov_list

class Training:
    """
        Training classe
    """

    def __init__(self, model, device, train_set, valid_set, opt, loss,
                  target='gt_mask', loss_weights=None, train_name='net', logger=None,
                  arch='unet', train_with_targets=False):
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
        if type(target) == list:
            self.target = target
        else:
            self.target = [target]
        self.n_out = len(self.target)
        if loss_weights == None:
            if self.n_out == 3:
                self.loss_weights = [.5, 0.25, 0.25]
            elif self.n_out == 2:
                self.loss_weights = [.75, 0.25]
            else:
                self.loss_weights = 1.
        else:
            self.loss_weights = 1.
        self.arch = arch
        self.train_with_targets = train_with_targets

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
        self.model = self.model.to(self.device)

        # Batch iteration - Training dataset
        for batch_idx, sample in enumerate(data_loader_train):
            # desired parameters
            pred_masks = None
            # output targets
            targets = []
            # Treat output
            if type(sample) is list: # list output
                bs = len(sample)
                ch, h, w = sample[0]['image'].shape
                # Get images
                image = torch.zeros(bs,ch, h, w)
                for i in range(bs):
                    image[i] = sample[i]['image']
                # Get masks
                for tgt_str in self.target:
                    targets.append([s[tgt_str] for s in sample])
            else:                   # Dict output
                # Load data
                image = sample['image'].to(self.device)
                # Get masks
                for tgt_str in self.target:
                    targets.append(sample[tgt_str].to(self.device))

                # Handle input
                if len(image.size()) < 4:
                    image.unsqueeze_(1) # add a dimension to the tensor

            # Run prediction
            if self.train_with_targets:
                loss_dict = self.model(image, targets[0])
                loss = self.criterion(loss_dict)
            else:
                pred_masks = self.model(image)
            
                # Handle multiples outputs
                if type(pred_masks) is list:
                    prediction = pred_masks[0]
                    losses = []
                    for k in range(len(pred_masks)):
                        losses.append(self.criterion(pred_masks[k], targets[k]) * self.loss_weights[k])
                    loss = sum(losses)
                else:
                    prediction = pred_masks
                    # Calculate loss for each batch
                    loss = self.criterion(pred_masks, targets[0])

            # Update epoch loss
            loss_train_sum += len(image) * loss.item()

            # Update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Output preview
            if batch_idx == len(data_loader_train) - 1:
                ref_image_train = image[0,...]
                ref_pred_train = prediction[0,...]

        # Calculate average loss per epoch
        avg_loss_train = loss_train_sum / data_train_len

        return avg_loss_train, ref_image_train, ref_pred_train


    def _iterate_val(self, data_loader_val):

        # Init loss count
        loss_val_sum = 0
        data_val_len = len(self.dataset_val)

        # To evaluate on validation set
        self.model.eval()
        self.model = self.model.to(self.device)

        # Batch iteration - Validation dataset
        for batch_idx, sample in enumerate(data_loader_val):
            # Load data
            image = sample['image'].to(self.device)
            gt_mask = sample['gt_mask'].to(self.device)

            # Handle input
            if len(image.size()) < 4:
                image.unsqueeze_(1) # add a dimension to the tensor

            # Prediction
            self.optimizer.zero_grad()
            pred = self.model(image)
            # Handle multiples outputs
            if type(pred) is list:
                pred = pred[0]

            # Calculate loss for each batch
            val_loss = self.criterion(pred, gt_mask)
            loss_val_sum += len(image) * val_loss.item()

            # Print output preview
            if batch_idx == len(data_loader_val) - 1:
                ref_image_val = image[0,...]
                ref_pred_val = pred[0,...]

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
        if self.train_with_targets:
            data_loader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_ov_list)
        else:
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
            print('')

            # ======================== Save weights ============================ #
            if best_loss > avg_loss_val:
                best_loss = avg_loss_val
                # save
                self._saveweights({
                            'epoch': epoch + 1,
                            'arch': self.arch,
                            'n_input': ref_image_train.shape[0],
                            'n_classes': ref_pred_train.shape[0],
                            'state_dict': self.model.state_dict(),
                            'target': self.target,
                            'loss_function': str(self.criterion),
                            'loss_weights': self.loss_weights,
                            'best_loss': best_loss,
                            'optimizer': str(self.optimizer),
                            'optimizer_dict': self.optimizer.state_dict(),
                            'device': str(self.device)
                            })

            # ====================== Tensorboard Logging ======================= #
            if self.logger:
                self._logging(epoch, avg_loss_train, avg_loss_val,
                    ref_image_train, ref_pred_train, ref_image_val, ref_pred_val)