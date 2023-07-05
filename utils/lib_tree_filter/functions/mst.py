import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import sys
sys.path.insert(0,"../")
import tree_filter_cuda as _C

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None

mst = _MST.apply


