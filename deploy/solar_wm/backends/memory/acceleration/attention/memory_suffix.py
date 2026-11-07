import torch
from solar.backends.minimax_h3.torch_prope import _project

def suffix(x, matrix):
    return torch.cat((x[..., :96], _project(x[..., 96:128], matrix=matrix)), dim=-1)
