import numpy as np
import torch
import torch.nn as nn
import add_op


class AddOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, input1, input2):
        outputs = add_op.forward(input1, input2)
        # ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(self, grad_output):
        # print(grad_output)
        output = add_op.backward(grad_output)
        return output, output  

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

    def forward(self, input1, input2):
        return AddOpFunction.apply(input1, input2)

def T(x, cuda=False):
    if not cuda:
        return torch.tensor(x, requires_grad=True)
    else:
        return torch.tensor(x, dtype=torch.float, device="cuda", requires_grad=True)

USE_CUDA = 0
model = MyNetwork()
if USE_CUDA:
    model.to('cuda')

x = np.arange(25, dtype=np.float32).reshape((5,5))
x2 = x * 2
input1, input2 = T(x, USE_CUDA), T(x2, USE_CUDA)

pred = model(input1, input2 * 2)
print(pred)
# gt = input1 + input2 
print("Backward...")
pred.sum().backward()
print(input1.grad)
print(input2.grad)
# gt.sum().backward()
