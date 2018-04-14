import warnings

from torch.autograd import Variable
import torch

class _Loss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
