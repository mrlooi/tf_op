import torch
from torch.autograd import Function
from _ext import add_op as my_lib


class MyAddFunction(Function):
    def forward(self, input1, input2):
        output = input1.new(*input1.size()).zero_()
        if input1.is_cuda:
            N = input1.nelement()
            my_lib.add_forward_cuda(N, input1, input2, output)
        else:
            my_lib.add_forward(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = torch.FloatTensor()
        my_lib.add_backward(grad_output, grad_input)
        return grad_input
