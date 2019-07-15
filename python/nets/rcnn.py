# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:30:06 2019

@author: Diego Wanderley
@python: 3.6
@description: Methods based on R-CNN for Object Detection, Instance Segmentation and Person Keypoint Detection

"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from nets.modules import *


class FasterRCNN(nn.Module):
    '''
    Faster R-CNN Class
    '''
    def __init__(self, n_channels=3, n_classes=21, softmax_out=False,
                        resnet_type=101, pretrained=False):
        super(FasterRCNN, self).__init__()

        self.resnet_type = resnet_type
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = None
        if n_channels != 3:
            self.inconv = FwdConv(n_channels, 3, kernel_size=1, padding=0)
        # Pre-trained model needs to be an identical network
        if pretrained:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=91, min_size=512)
            # Reset output
            if n_classes != 91:
                self.body.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
                self.body.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4*n_classes, bias=True)

        else:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=n_classes, min_size=512)

        # Softmax alternative
        self.has_softmax = softmax_out
        if softmax_out:
            self.softmax = nn.Softmax2d()
        else:
            self.softmax = None

    def forward(self, x, tgts=None):
        # input layer to convert to 3 channels
        if self.inconv != None:
            x = self.inconv(x)
        # Verify if is traning (this situation requires targets)
        if self.body.training:
            x = list(im for im in x) # convert to list (as required)
            x_out = self.body(x,tgts)
        else:
            x_out = self.body(x)
        # Softmax output if necessary
        if self.softmax != None:
            x_out = self.softmax(x_out)
        return x_out


class MaskRCNN(nn.Module):
    '''
    Mask R-CNN Class
    '''
    def __init__(self, n_channels=3, n_classes=21, softmax_out=False,
                        resnet_type=101, pretrained=False):
        super(MaskRCNN, self).__init__()

        self.resnet_type = resnet_type
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pretrained = pretrained

         # Input conv is applied to convert the input to 3 ch depth
        self.inconv = None
        if n_channels != 3:
            self.inconv = FwdConv(n_channels, 3, kernel_size=1, padding=0)
        # Pre-trained model needs to be an identical network
        if pretrained:
            self.body = maskrcnn_resnet50_fpn(pretrained=pretrained, num_classes=91, min_size=512)
            # Reset output
            if n_classes != 91:
                self.body.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
                self.body.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4*n_classes, bias=True)

                self.body.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))

        else:
            self.body = fasterrcnn_resnet50_fpn(pretrained=pretrained, num_classes=n_classes, min_size=512)

        # Softmax alternative
        self.has_softmax = softmax_out
        if softmax_out:
            self.softmax = nn.Softmax2d()
        else:
            self.softmax = None

    def get_output_segmentation(self, x):
        '''
        Calculate output segmentation given list of dictionaries with masks and labels.
        '''
        x_out = []
        for el in x:
            insts, _, h, w = el['masks'].shape
            x_temp = torch.zeros(1, self.n_classes, h, w)
            #
            for lbl, mask in zip(el['labels'], el['masks']):
                idx = lbl.item()
                x_temp[:,idx,...] = x_temp[:,idx,...] + mask
            x_temp[:,0,...] = torch.ones(1, 1, h, w) - x_temp[:,1,...] - x_temp[:,2,...]
            # Threshold from 0 to 1
            x_temp = torch.clamp(x_temp, 0., 1.)
            # Add to output
            x_out.append(x_temp)

        return x_out

    def forward(self, x, tgts=None):
        if self.inconv != None:
            x = self.inconv(x)
        # Verify if is traning (this situation requires targets)
        if self.body.training:
            x = list(im for im in x) # convert to list (as required)
            x_out = self.body(x,tgts)
        else:
            x_out = self.body(x,tgts)
        if self.softmax != None:
            x_out = self.softmax(x_out)
        return x_out


import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


if __name__ == "__main__":

    import math

    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together
    # https://github.com/pytorch/vision/blob/master/references/detection/train.py


    # Images
    images = torch.randn(2, 1, 512, 512)
    # Targets
    bbox = torch.FloatTensor([[120, 130, 300, 350], [200, 200, 250, 250]]) # [y1, x1, y2, x2] format
    lbls = torch.LongTensor([1, 2]) # 0 represents background
    mask = torch.zeros(2, 512, 512)
    mask[0, 120:300, 130:350] = 1
    mask[1, 200:250, 200:250] = 1
    # targets per image
    tgts = {
        'boxes':  bbox,
        'labels': lbls,
        'masks': mask
    }
    # targets to list (by batch)
    targets = [tgts, tgts]
    #images = list(image for image in images)

    # Model
    model = MaskRCNN(n_channels=1, n_classes=3, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #model.eval()
    model.train()

    # output
    loss_dict = model(images, targets)

    if model.training:
        # multi-task loss
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Update weights
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    else:
        model.get_output_segmentation(loss_dict)

    print(loss_dict)
