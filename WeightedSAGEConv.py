import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
import datetime
from haversine import haversine, Unit
import argparse
import pickle

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing

#The weighted Graph Sage network, copied from PyG GraphSage code and GCN code
class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(WeightedSAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                normalize: Tensor) -> Tensor:
        
        x: OptPairTensor = (x, x)
   
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, norm = normalize)
        out = self.lin_l(out)

        x_r = x[1]
        out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, norm) -> Tensor:
        return x_j*norm.view(-1,1)