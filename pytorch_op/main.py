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

model = MyNetwork().cuda()

x = np.arange(25).reshape((5,5))
x2 = x * 2
input1, input2 = torch.FloatTensor(x), torch.FloatTensor(x2)
print(model(input1.cuda(), input2.cuda()))
print(input1 + input2)
