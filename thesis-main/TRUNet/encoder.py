# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import torch.nn as nn
from torch.nn import LayerNorm

from block import Block


# from An image is worth 16x16 words:
# The Transformer encoder (Vaswani et al., 2017) consists of alternating
# layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3).


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList() #Anmar:Holds submodules in a list
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)


        # make the layers within the transformer
        for _ in range(config.transformer_num_layers):
            # block contains multi-head attention and mlp
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer)) #Anmar: Appends to the module list. 

    def forward(self, hidden_states): 
        #print("forward method Encoder class")
        attn_weights = []
        for layer_block in self.layer:
            #Anmar: This is the number of layers we use
            hidden_states = layer_block(hidden_states) #Anmar: Forward() for block function

            
        encoded = self.encoder_norm(hidden_states) 
        return encoded
