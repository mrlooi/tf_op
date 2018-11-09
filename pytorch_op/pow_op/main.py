import numpy as np
import torch
from modules.pow import PowModule
from functions.pow import PowFunction

def FT(x):
	return torch.tensor(x, dtype=torch.float32, requires_grad=True)

x = np.array([4,3])
x2 = np.array([2,3])
input1, input2 = FT(x), FT(x2)

pow_op = PowFunction()
x = pow_op(input1, input2)
x.sum().backward()
# print(torch.pow(input1, input2))
# pow_op.backward()
# z = PowFunction.apply(input1, input2)
