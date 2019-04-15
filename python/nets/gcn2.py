import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math
from nets.modules import *


class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
    
    def forward(self,x):
        x_res = x
#         x_res = self.bn(x)
#         x_res = self.relu(x_res)
        x_res = self.conv1(x_res)
#         x_res = self.bn(x_res)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        
        x = x + x_res
        
        return x


class GCN_block(nn.Module):
    def __init__(self,c,out_c,k=7): #out_Channel=21 in paper
        super(GCN_block, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k,1), padding =((k-1)/2,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k), padding =(0,(k-1)/2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k), padding =((k-1)/2,0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k,1), padding =(0,(k-1)/2))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x


class bottleneck_GCN(nn.Module):
    def __init__(self, m, c, k):
        super(bottleneck_GCN, self).__init__()
        self.bn_m = nn.BatchNorm2d(m)
        self.bn_c = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_l1 = nn.Conv2d(c, m, kernel_size=(k,1), padding =((k-1)/2,0))
        self.conv_l2 = nn.Conv2d(m, m, kernel_size=(1,k), padding =(0,(k-1)/2))
        self.conv_r1 = nn.Conv2d(c, m, kernel_size=(1,k), padding =((k-1)/2,0))
        self.conv_r2 = nn.Conv2d(m, m, kernel_size=(k,1), padding =(0,(k-1)/2))
        self.conv_f = n.Con2vd(m, c, kernel_size=1, padding=0)
        
    def forward(self,x):
        x_res_l = self.conv_l1(x)
        x_res_l = self.bn_m(x_res_l)
        x_res_l = self.relu(x_res_l)
        x_res_l = self.conv_l2(x_res_l)
        x_res_l = self.bn_m(x_res_l)
        x_res_l = self.relu(x_res_l)
        
        x_res_r = self.conv_r1(x)
        x_res_r = self.bn_m(x_res_r)
        x_res_r = self.relu(x_res_r)
        x_res_r = self.conv_r2(x_res_r)
        x_res_r = self.bn_m(x_res_r)
        x_res_r = self.relu(x_res_r)
        
        x_res = x_res_l + x_res_r
        x_res = self.conv_f(x_res)
        x_res = self.bn_c(x_res)
        
        x = x + x_res
        
        return x


class FCN_GCN(nn.Module):
    def __init__(self, n_channels, num_classes):
        self.num_classes = num_classes #21 in paper
        super(FCN_GCN, self).__init__()
        
        # Input conv is applied to convert the input to 3 ch depth
        self.inconv = fwdconv(n_channels, 3, kernel_size=1, padding=0)
        
        resnet = models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN_block(256,self.num_classes,55) #gcn_i after layer-1
        self.gnc2 = GCN_block(512,self.num_classes,27)
        self.gcn3 = GCN_block(1024,self.num_classes,13)
        self.gcn4 = GCN_block(2048,self.num_classes,7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)
        self.br9 = BR(num_classes)

    def _classifier(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c,in_c,3,padding=1,bias=False),
            nn.BatchNorm2d(in_c/2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(in_c/2, self.num_classes, 1),

            )    

    def forward(self,x):
        input = x
        x = self.inconv(x)
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        convA_x = x
        x = self.maxpool(x)
        pooled_x = x

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        gc_fm4 = F.Upsample(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.Upsample(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.Upsample(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.Upsample(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)

        gc_fm1 = F.Upsample(self.br8(gc_fm1), scale_factor=2, mode='bilinear', align_corners=True)

        out = F.Upsample(self.br9(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Main calls
if __name__ == '__main__':
    model_bas = FCN_GCN(1,1)

    print(model_bas)