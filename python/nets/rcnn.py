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
from modules import *


class FasterRCNN(nn.Module):
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

    def forward(self, x):
        if self.inconv != None:
            x = self.inconv(x)
        x_out = self.body(x)
        if self.softmax != None:
            x_out = self.softmax(x_out)
        return x_out


class MaskRCNN(nn.Module):
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

    def forward(self, x):
        if self.inconv != None:
            x = self.inconv(x)
        x_out = self.body(x)
        if self.softmax != None:
            x_out = self.softmax(x_out)
        return x_out



if __name__ == "__main__":

    image = torch.randn(4, 1, 512, 512)

    bbox = torch.FloatTensor([[120, 130, 300, 350], [200, 200, 250, 250]]) # [y1, x1, y2, x2] format
    labels = torch.LongTensor([1, 2]) # 0 represents background
    sub_sample = 16


    model = FasterRCNN(n_channels=1, n_classes=3, pretrained=True)
    #model.eval()
    model.train()

    output = model(image)
    #ValueError: In training mode, targets should be passed

    print(output)

    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
    loss_temp += loss.item()
