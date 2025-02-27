import torch
import cv2
import imageio
import torch.nn.functional as F
import torch.nn as nn
from lib.modules import *
from torchvision.transforms.functional import rgb_to_grayscale
from lib.HFE import HFE
from lib.pvtv2 import pvt_v2_b2
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class HFANet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(HFANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.backbone = pvt_v2_b2()
        path = '/DATA/home/zyw/LJH/LSSNet-main/LSSNet-main/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.CBR5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.CBR4 = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.CBR3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.CBR2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.up5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) 
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.hFA1 = HFA(64)
        self.hFA2 = HFA(64)
        self.hFA3 = HFA(128)
        self.hFA4 = HFA(256)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

    def forward(self, x):
        grayscale_img = rgb_to_grayscale(x)
        high_information = HFE(grayscale_img, 5, 1)

        high_information = high_information[1]
        high_information = torch.clamp(high_information,0,255)
        high_information = (high_information / 255.0) ** 1.4 * 255.0
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        f2, f3, f4, f5 = self.backbone(x)
        f_c5 = self.CBR5(f5) 
        f_c4 = self.CBR4(f4)
        f_c3 = self.CBR3(f3)
        f_c2 = self.CBR2(f2)

        #Decoder
        f_d5 = self.up5(f_c5)
        out5 = self.out5(f_d5) 
        hfa4 = self.hFA4(high_information, f_c4, out5)
        f_d4 = self.up4(f_d5, hfa4)
        out4 = self.out4(f_d4)    
        hfa3 = self.hFA3(high_information, f_c3, out4)
        f_d3 = self.up3(f_d4, hfa3)
        out3 = self.out3(f_d3)
        hfa2 = self.hFA2(high_information, f_c2, out3)
        f_d2 = self.up2(f_d3, hfa2)
        out2 = self.out2(f_d2)  
        return  out2, out3, out4, out5


class HFEANetModel(nn.Module): 
    def __init__(self, n_channels=3, n_classes=1):
        super(HFEANetModel,self).__init__()
        self.channel = n_channels
        self.num_classes = n_classes
        self.net = HFANet(self.channel, self.num_classes)

    def forward(self, images):
        out2, out3, out4, out5= self.net(images)
        return out2, out3, out4, out5  
