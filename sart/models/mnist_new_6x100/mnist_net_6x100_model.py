from torch import nn
from torch.nn import  Sequential, Dropout

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = Sequential(
            nn.Linear(784, 100),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(100, 100),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(100, 100),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(100, 100),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(100, 100),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(100, 100),
            # Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x