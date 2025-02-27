import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# 定义 Sobel 卷积核
def HFE_kernel(channels=1, cuda=True):
    kernel_x = torch.tensor([[1., 0., -1.],
                              [2., 0., -2.],
                              [1., 0., -1.]])
    
    kernel_y = torch.tensor([[1., 2., 1.],
                              [0., 0., 0.],
                              [-1., -2., -1.]])

    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    
    if cuda:
        kernel_x = kernel_x.cuda(0)
        kernel_y = kernel_y.cuda(0)
    
    return kernel_x, kernel_y

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_kernel(img, kernel_x, kernel_y):
    img = F.pad(img, (1, 1, 1, 1), mode='reflect')
    out_x = F.conv2d(img, kernel_x, groups=img.shape[1])
    out_y = F.conv2d(img, kernel_y, groups=img.shape[1])
    return out_x, out_y

def upsample1(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return x_up

def HFE(img, level, channels):
    current = img
    pyr = []
    
    for _ in range(level):
        kernel_x, kernel_y = HFE_kernel(channels)
        sobel_x, sobel_y = conv_kernel(current, kernel_x, kernel_y)
        gradient_magnitude = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        down = downsample(gradient_magnitude)
        up = upsample1(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = up
    pyr.append(current)
    return pyr

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample2(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def edge_information(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample2(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff
