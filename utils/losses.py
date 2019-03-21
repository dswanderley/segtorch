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


    def _sort_instances(self, pred, gt):

        # Get unic labesl from 0 to max (n_instances-1)
        unique_labels = torch.unique(gt, sorted=True) # instances labels (including background = 0)
        # Get data dimensions
        n_instances = len(unique_labels)
        n_filters, n_loc = pred.size()
        # Reshape and expand (repeat) to the number of instances
        pred_repeated = pred.unsqueeze(2).expand(n_filters, n_loc, n_instances).contiguous()  # n_filters, n_loc, n_instances

        # Mask with instances, each depth is a instance
        imasks = torch.zeros(n_loc, n_instances)
        for i in range(n_instances):
            imasks[...,i] = torch.where(gt == unique_labels[i], torch.ones(gt.shape), torch.zeros(gt.shape))
        # Reshape and expand (repeat) to the number of features
        imasks.unsqueeze_(0) # 1, n_loc, n_instances
        
        return pred_repeated, imasks, n_instances
    

    def _variance_term(self, mu, x, gt):
        ''' l_var  - intra-cluster distance '''

        # Count pixels of each instance
        counts = torch.sum(gt, dim=1)
        _, C = counts.shape # number of clusterss
        
        # Mean of the instance at each expected position for that instance
        mu_expand = mu.unsqueeze_(1).expand(x.shape) * gt
        
        # Calculate intra distance
        diff = torch.norm(mu_expand - x, dim=0)
        distance = torch.clamp(diff - self.delta_v, 0., 100000.)**2

        # variance
        l_var = torch.sum(torch.div(torch.sum(distance), counts)) / C

        return l_var


    def _distance_term(self, mu):
        ''' l_dist - inter-cluster distance'''

        # number features and number of clusterss
        nf, _, C = mu.shape

        # Prepare data - meshgrid
        means = mu.reshape(nf,C).permute(1, 0)
        means_1 = means.unsqueeze(1).expand(C, C, nf)
        means_2 = means_1.permute(1, 0, 2)
        
        # Calculate norm of distance
        diff = means_1 - means_2 
        norm = torch.norm(diff, dim=2)
        margin = Variable(2 * self.delta_d * (1.0 - torch.eye(C))) # cluster radius

        # calculate distance term
        l_dist = torch.sum(torch.clamp(margin - norm, 0., 100000.)**2) / (C*(C-1))

        return l_dist


    def _regularization_term(self, mu, num_instances):
        ''' l_reg - regularization term '''

        l_reg = (torch.norm(mu, dim=0) / num_instances).sum()

        return l_reg


    def _discriminative_loss(self, pred, tgt):

        # Adjust data - CHECK IF NECESSARY
        correct_label = tgt.unsqueeze_(0).view(1, self.height * self.width)
        correct_label.long()

        # Prediction
        reshaped_pred = pred.reshape(self.n_features, self.height*self.width).contiguous()

        # Count instances
        pred_repeated, unique_id, num_instances = self._sort_instances(reshaped_pred, correct_label)

        # Calculate correspondence map of prediction
        pred_masked = pred_repeated * unique_id
        # Calculate means
        means = torch.div(pred_masked.sum(1), unique_id.sum(1))

        # Variance term
        l_var = self._variance_term(means, pred_masked, unique_id)
        # Distance term
        l_dist = self._distance_term(means)
        # Regularization term
        l_reg = self._regularization_term(means, num_instances)

        # Loss
        loss = self.alpha * l_var #+ self.beta *  l_dist + self.gamma * l_reg
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
