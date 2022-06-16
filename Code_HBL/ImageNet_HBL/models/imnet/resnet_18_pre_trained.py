
from __future__ import absolute_import

__all__ = ['resnet']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import random
from pylab import *
import os
import math

import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision
from torchvision import datasets, models, transforms


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18(nn.Module):

    def __init__(self, output_dims=10, polars=None):
        super(ResNet18, self).__init__()
        print('ResNet 18 pretrained is being loaded.')
        self.polars = polars

        ###################################################################################### Load pre-trained model and freeze layers
        self.model = torch_models.resnet18(pretrained=True)

        # Freeze all layers except the first and the last one
        # self.ct = 0
        # for child in self.model.children(): # Include the first layer for learning as well
        #     self.ct += 1
        #     if self.ct > 1:
        #         for param in child.parameters():
        #             param.requires_grad = False

        # ####### old - if only output layer is learnable
        for param in self.model.parameters():
            param.requires_grad = False
        # ####### end old
        
        # # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_dims)
        #####################################################################################

        ###################################################################################### Load pre-trained model, use initialised weight but allow learning
        # self.model = torch_models.resnet18(pretrained=True)

        ####### old - if only output layer is learnable
        # for param in self.model.parameters():
        #     param.requires_grad = False
        ####### end old
        
        # Parameters of newly constructed modules have requires_grad=True by default
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, output_dims)
        #####################################################################################


    # def get_model(output_dims = 10):

    #     model = torch_models.resnet18(pretrained=True) # Option 2
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     # Parameters of newly constructed modules have requires_grad=True by default
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, output_dims)
    #     print('hello')
    #     return model


    def forward(self, x):
        x = self.model(x)
        return x



    def predict(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = torch.mm(x, self.polars.t().cuda())
        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet18(**kwargs)

