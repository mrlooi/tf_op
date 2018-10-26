import numpy as np
import torch
import torch.nn as nn
from modules.add import MyAddModule

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)

USE_CUDA = 1
model = MyNetwork()
if USE_CUDA:
    model.cuda()

x = np.arange(25).reshape((5,5))
x2 = x * 2
input1, input2 = torch.FloatTensor(x), torch.FloatTensor(x2)

if USE_CUDA:
    input1 = input1.cuda()
    input2 = input2.cuda()
print(model(input1, input2))
print(input1 + input2)
