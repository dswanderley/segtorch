# -*- coding: utf-8 -*-
"""
Created on Wed Fev 27 18:37:58 2019

@author: Diego Wanderley
@python: 3.6
@description: Alternative Loss Functions (Dice)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class DiceLoss(nn.Module):
    '''
    Dice Loss (Ignore background - channel 0)

    Arguments:
        @param prediction: tensor with predictions classes
        @param groundtruth: tensor with ground truth mask
    '''

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward (self, pred, gt):

        SMOOTH = 0.0001

        # Ignorne background
        prediction = pred[:,1:,...].contiguous()
        groundtruth = gt[:,1:,...].contiguous()

        iflat = prediction.view(-1)
        tflat = groundtruth.view(-1)

        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()
        dsc = ((2. * intersection + SMOOTH) / (union + SMOOTH))

        loss_dsc = 1. - dsc
        return loss_dsc


class DiceCoefficients(nn.Module):

    def __init__(self):
        super(DiceCoefficients, self).__init__()

    def forward (self, pred, target):

        SMOOTH = 0.0001

        nclasses = target.size()[1]
        dsc = []

        for i in range(nclasses):

            # Ignorne background
            prediction = pred[:,i,...].contiguous()
            groundtruth = target[:,i,...].contiguous()

            iflat = prediction.view(-1)
            tflat = groundtruth.view(-1)

            intersection = (iflat * tflat).sum()
            union = iflat.sum() + tflat.sum()
            dsc.append((2. * intersection + SMOOTH) / (union + SMOOTH))

        return dsc


class DiscriminativeLoss(nn.Module):
    """
        Discriminative Loss function
    """
    def __init__(self, n_features, delta_v=0.5, delta_d=1.5, alpha = 1., beta = 1., gamma = 0.001):
        super(DiscriminativeLoss, self).__init__()
        
        self.n_features = n_features
        self.delta_v = delta_v 
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    

    def _sort_instances(self, correct_label, reshaped_pred):
        
        # Count instances
        unique_labels = torch.unique(correct_label, sorted=True) # instances labels (including background = 0)
        num_instances  = len(unique_labels) # number of instances (including background)
        counts = torch.histc(correct_label.float(), bins=num_instances, min=0, max=num_instances-1)
        counts = counts.expand(self.n_features, num_instances)   # expected amount of pixel for each instance
        unique_id = correct_label.expand(self.n_features, self.height * self.width).long() # expected index of each pixel
        
         # Get sum by instance
        segmented_sum = torch.zeros(self.n_features, num_instances).scatter_add_(1, unique_id, reshaped_pred)
        # Mean of each instance in each feature layer
        mu = torch.div(segmented_sum, counts)

        return num_instances, counts, unique_id, mu


    def _variance_term(self, mu, num_instances, unique_id, counts, reshaped_pred):
        ''' l_var  - intra-cluster distance '''

        # Mean of the instance at each expected position for that instance
        mu_expand = torch.gather(mu, 1, unique_id)

        # Calculate intra distance
        distance = torch.clamp(torch.norm(mu_expand - reshaped_pred, dim=0) - self.delta_v, 0., 10000)**2 # max(0,x)   # apply delta_v
        distance.reshape(1,len(distance)).contiguous()
        
        l_var = torch.zeros(1, num_instances).scatter_add_(1, unique_id[0].reshape(1, self.height * self.width), distance.reshape(1, self.height * self.width))
        l_var /= counts / num_instances
        print(id(l_var))
        #print(l_var)

        return l_var.sum()
        
    
    def _distance_term(self, mu, num_instances):
        ''' l_dist - inter-cluster distance'''

        # Calculate inter distance
        mu_sdim = mu.reshape(mu.shape[1] * mu.shape[0]) # reshape to apply meshgrid
        mu_x, mu_y = torch.meshgrid(mu_sdim, mu_sdim)
        aux_x = mu_x[:,:num_instances].reshape(self.n_features, num_instances, num_instances)
        aux_y = mu_y[:num_instances, :].reshape(num_instances, self.n_features, num_instances).permute(1,0,2)
        # Calculate differece interclasses
        mu_diff = aux_x - aux_y
        mu_diff = torch.norm(mu_diff,dim=0)
        # Use a matrix with delt_d to calculate each difference
        aux_delta_d = 2 * self.delta_d * (torch.ones(mu_diff.shape) - torch.eye(mu_diff.shape[0])) # ignore diagonal (C_a = C_b)
        aux_delta_d = Variable(aux_delta_d)
        l_dist = torch.clamp(aux_delta_d - mu_diff, 0., 10000)**2 # max(0,x)
        # 1 / C(C-1)
        l_dist /= num_instances / (num_instances - 1)
        #print(l_dist)

        return l_dist.sum() 


    def _regularization_term(self, mu, num_instances):
        ''' l_reg - regularization term '''

        l_reg = torch.norm(mu, dim=0) / num_instances
        #print(l_reg)
        
        return l_reg.sum()


    def _discriminative_loss(self, pred, tgt):

        # Adjust data - CHECK IF NECESSARY
        correct_label = tgt.unsqueeze_(0).view(1, self.height * self.width)
        correct_label.long()

        # Prediction
        reshaped_pred = pred.reshape(self.n_features, self.height*self.width).contiguous()

        # Count instances
        num_instances, counts, unique_id, mu = self._sort_instances(correct_label, reshaped_pred)
        
        # Variance term
        l_var = self._variance_term(mu, num_instances, unique_id, counts[0], reshaped_pred)
        print(id(l_var))
        # Distance term
        l_dist = self._distance_term(mu, num_instances)
        # Regularization term
        l_reg = self._regularization_term(mu, num_instances)

        # Loss
        loss = self.alpha * l_var + self.beta *  l_dist + self.gamma * l_reg
        #print(loss)

        return loss, l_var, l_dist, l_reg


    def forward(self, prediction, target):

        # Adjust data - CHECK IF NECESSARY
        self.batch_size, self.height, self.width = target.shape

        loss_list = []

        for i in range(self.batch_size):

            pred = prediction[i,...].contiguous()
            tgt = target[i,...].contiguous()

            loss, l_var, l_dist, l_reg = self._discriminative_loss(pred, tgt)

            loss_list.append(loss)

        out_loss = torch.stack(loss_list)
        out_loss = torch.sum(out_loss)
                        
        return  out_loss
                