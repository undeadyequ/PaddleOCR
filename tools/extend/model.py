import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch

def modified_resnet50():
    # load pretrained resnet50 with a modified last fully connected layer
    model = models.resnet50(pretrained=True)
    model.fc = FinalLayer()

    # uncomment following lines if you wnat to freeze early layers
    # i = 0
    # for child in model.children():
    #     i += 1
    #     if i < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False

    return model


def vis_model():
    model = models.resnet50(pretrained=True)
    return model


def det_model():
    return 1


def detrec_model():
    return 1


class JointVisDet(nn.Module):
    def __init__(self, idim=1003, odim=12):
        self.vis_model = vis_model()
        self.det_model = det_model()
        self.head = torch.nn.Linear(idim, odim)

    def forward(self, img):
        vis_out = self.vis_model(img)
        det_out = self.det_model(img)
        jot_out = torch.cat((vis_out, det_out), 1)
        out = self.head(jot_out)
        return out


class JointVisDetFineGrained(nn.Module):
    def __init__(self, idim=1512, odim=5):
        self.vis_model = vis_model()
        self.detrec_model = detrec_model()
        self.head = torch.nn.Linear(idim, odim)

    def forward(self, img):
        vis_out = self.vis_model(img)
        detrec_out = self.detrec_model(img)
        jot_out = torch.cat((vis_out, detrec_out), 1)
        out = self.head(jot_out)
        return out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class FinalLayer(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out
