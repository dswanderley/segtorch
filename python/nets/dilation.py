import torch
import torch.nn as nn
from torchvision import models
from nets.modules import *


class CAN(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(CAN, self).__init__()
        # parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        # net
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True)
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0, bias=True, dilation=4), #fc6 layer
            nn.ReLU(inplace=True)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=True), #fc7 layer
            nn.ReLU(inplace=True)
            )
        self.layer8 = nn.Sequential(
            nn.Conv2d(4096, n_classes, kernel_size=1, stride=1, padding=0, bias=True), #final layer
            nn.ReLU(inplace=True)
            )
        self.layer9 = nn.Sequential(
            nn.ZeroPad2d(1),

            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True), #ctx_conv
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
            )
        self.layer10 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True)
            )
        self.layer11 = nn.Sequential(
            nn.ZeroPad2d(4),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True, dilation=4),
            nn.ReLU(inplace=True)
            )
        self.layer12 = nn.Sequential(
            nn.ZeroPad2d(8),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True, dilation=8),
            nn.ReLU(inplace=True)
            )
        self.layer13 = nn.Sequential(
            nn.ZeroPad2d(16),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True, dilation=16),
            nn.ReLU(inplace=True)
            )
        self.layer14 = nn.Sequential(
            nn.ZeroPad2d(32),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True, dilation=32),
            nn.ReLU(inplace=True)
            )
        self.layer15 = nn.Sequential(
            nn.ZeroPad2d(64),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True, dilation=64),
            nn.ReLU(inplace=True)
            )
        self.layer16 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            # Upsampling needs
        self.layer18 = nn.Sequential(
            nn.Conv2d(n_classes, n_classes, kernel_size=16, stride=1, padding=7, bias=False),
            nn.ReLU(inplace=True)
            )
        self.softmax = nn.Sequential(
             nn.Softmax(dim=1)
            )


    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        x17 = nn.functional.interpolate(x16, size=(512+1,512+1), mode='bilinear', align_corners=True)
        x18 = self.layer18(x17)

        x_out = self.softmax(x18)
        return x_out

# Main calls
if __name__ == '__main__':
    #model_bas = FCN_GCN(1,1)

    model_bas = CAN(1,3)
    #print(model_bas)

    x = torch.randn(1, 1, 512, 512)
    #print(x)

    y = model_bas(x)
    print(y)