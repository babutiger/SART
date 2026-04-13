from cmath import sqrt
from torch import nn
import numpy as np
import torch
import numpy as np
import math

# def softplus(x):
#     if x >= 20.0:
#         x = torch.tensor(x)
#         return x
#     else:
#         x = torch.tensor(x)
#         x = 0.5 * (x + torch.sqrt(torch.square(x) + 0.2))
#         return x


def softplus(x):
    input = torch.tensor(x)
    output = nn.Softplus()(input)
    return output

def gelu(x):
    input = torch.tensor(x)
    output = nn.GELU()(input)
    return output




if __name__ == '__main__':
    ans1 = softplus(1.0)
    ans2 = gelu(-2.0)

    ans3 = softplus(0.0)
    ans4 = gelu(-1.0)

    ans5 = np.log(1+(1+math.exp( 2 ))*(1+math.exp( 2 )))

    print(ans1)
    print(ans2)
    print(ans3)
    print(ans4)