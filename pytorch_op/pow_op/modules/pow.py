from torch.nn import Module
from functions.pow import PowFunction

class PowModule(Module):
    def __init__(self):
        super(PowModule, self).__init__()
        
    def forward(self, input1, input2):
        return PowFunction()(input1, input2)
