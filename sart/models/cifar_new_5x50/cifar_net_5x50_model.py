
from torch import nn
from torch.nn import  Sequential, Dropout

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = Sequential(
            nn.Linear(3072, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            
            nn.Linear(50, 10)
        )
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(-1, x.size(0))

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         # self.liner0 = torch.nn.Flatten()
#         self.linear1 = torch.nn.Linear(3072, 100)
#         self.linear2 = torch.nn.Linear(100, 100)
#         self.linear3 = torch.nn.Linear(100, 10)
#
#     def forward(self, x):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
#         # x = x.view(-1, x.size(0))
#         x = F.relu(self.linear1(x))
#
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear2(x))
#
#         x = self.linear3(x)
#         return x