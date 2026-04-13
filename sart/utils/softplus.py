from cmath import sqrt
from torch import nn
import numpy as np
import torch


# def softplus(x, b=0.01):
#     # return np.log(1 + np.exp(x))
#     x = 0.5 * (x + sqrt(x*x + b))
#     return x


def softplus(x):
    input = torch.tensor(x)
    output = nn.Softplus()(input)
    return output

def gelu(x):
    input = torch.tensor(x)
    output = nn.GELU()(input)
    return output




if __name__ == '__main__':
    ans1 = softplus(9.0)
    # ans2 = gelu(-2.0)

    print(ans1)