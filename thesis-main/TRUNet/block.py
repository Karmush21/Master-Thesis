# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from attention import Attention
from mlp import Mlp

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
        


    def forward(self, x, layer_num): #Anmar: Input size here will be [batch_size, 2744, 768]
        
        h = x
        x = self.attention_norm(x) #Anmar: Layernorm before attention, as in paper
        
        x = self.attn(x, layer_num) #Anmar: forward pass attnetion, output from this is same as input. 

        
        #attention_maps = attention_maps.cpu()
        #attention_maps = attention_maps.detach().numpy()

        x = x + h 

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x) #Anmar: forward pass Mlp
        
        x = x + h


        return x
