import numpy as np
import torch
from modules.pow import PowModule
from functions.pow import PowFunction

def FT(x):
	return torch.tensor(x, dtype=torch.float32, requires_grad=True)

x = np.array([4,3])
x2 = np.array([3,4])
input1, input2 = FT(x), FT(x2)

pow_op = PowFunction()
pow_m = PowModule()

# v1 = input1 * 2
x = pow_m(input1, input2)
x *= 3
# x = torch.pow(x, 2)
x.sum().backward()
print(input1.grad)
print(input2.grad)


input1.grad.zero_()
input2.grad.zero_()

# v1 = input1 * 2
x = torch.pow(input1, input2)
x *= 3
x.sum().backward()
print(input1.grad)
print(input2.grad)
