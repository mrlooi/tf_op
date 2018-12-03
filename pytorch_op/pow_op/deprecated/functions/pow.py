import torch
from torch.autograd import Function
from _ext import pow_op as my_lib


class PowFunction(Function):
    def __init__(self):
        self.input1 = None
        self.input2 = None

    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

        output = input1.new(*input1.size()).zero_()
        if input1.is_cuda:
            N = input1.nelement()
            my_lib.pow_forward_cuda(N, input1, input2, output)
        else:
            my_lib.pow_forward(input1, input2, output)
        return output

    def backward(self, grad_output):
        """
        In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
        """
        print("BACKWARD")
        print(grad_output)
        grad_input1 = self.input1.new(*self.input1.size()).zero_()
        grad_input2 = self.input2.new(*self.input2.size()).zero_()
        my_lib.pow_backward(grad_output, self.input1, self.input2, grad_input1, grad_input2)
        return grad_input1, grad_input2

