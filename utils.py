import os

import torch
from torch.autograd import Variable


"""Converts numpy to variable."""
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

"""Converts variable to numpy."""
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


"""Creates a directory if it does not already exist."""
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
