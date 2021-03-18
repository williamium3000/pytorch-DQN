import torch
from torch import nn
import random
import numpy as np
import torchvision
import torchvision.utils
import torch
import torch.nn as nn
from torchvision import models

class ravel(nn.Module):
    def __init__(self):
        super(ravel, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
class Q_network(nn.Module):
    def __init__(self, num_obs, num_act):
        super(Q_network, self).__init__()
        # features = (num_obs[0] // 8) * (num_obs[1] // 8) * 64
        self.backbone = nn.Sequential(

            # nn.Conv2d(num_obs[2], 32, 7, stride=1, padding=3),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(32, 64, 5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # ravel(),
            # nn.Linear(features, 64, True),
            # nn.ReLU(),
            # nn.Linear(64, num_act, True),

        )
        # self.backbone = models.resnet18(pretrained=True)
        # feature_extraction = False
        # if feature_extraction:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        # self.backbone.fc = nn.Linear(in_features = 512,  out_features = 256)
        # self.relu = nn.ReLU()
        # self.fc = nn.Linear(in_features = 256, out_features = num_act)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.relu(x)
        # x = self.fc(x)
        return x