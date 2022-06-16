
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class fullcon(nn.Module):

    def __init__(self, dims = 10, output_dims = 10, dr = 0.2, polars = None):
        super().__init__()

        self.polars = polars

        self.fc1 = nn.Linear(dims, 300) # Input Layer
        self.fc1_bn = nn.BatchNorm1d(num_features = 300, momentum=0.999)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(300, 300) # Hidden Layer 1
        self.fc2_bn = nn.BatchNorm1d(num_features = 300, momentum=0.999)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(300, 300) # Hidden Layer 2
        self.fc3_bn = nn.BatchNorm1d(num_features = 300, momentum=0.999)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(300, output_dims) # Output Layer
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = self.relu2(x)
        x = self.fc4(x)
        return x


    def predict(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = torch.mm(x, self.polars.t().cuda())
        return x





