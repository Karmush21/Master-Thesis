# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from functions import *

from pure_ViT_transformer import Pure_ViT_Transformer3d

from patch_embedding_block import PatchEmbeddingBlock

from pure_vit_decoder import pure_vit_decoder

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Pure_ViT(nn.Module): #Has to inhert from nn.module
    def __init__(self, config):
        #print("Initialize model in ViT.py")
        super().__init__()  # defining this as a model in nn
        
        self.config = config

        #TODO Make this less hard coded later
        self.patch_embedding = PatchEmbeddingBlock(in_channels = 1, 
                                        img_size = 224, 
                                        patch_size = 16, 
                                        hidden_size = self.config.hidden_size, 
                                        num_heads = self.config.transformer_num_heads, 
                                        proj_type = "perceptron", 
                                        pos_embed_type = "learnable", 
                                        dropout_rate = 0.0, 
                                        spatial_dims = 3)
        
        self.NEW_PURE_Transformer3d = Pure_ViT_Transformer3d(self.config)
        
        self.decoder = pure_vit_decoder(self.config)



    def forward(self, x):
       #Creates the patches and the matrix needed as input to transformer
       x = self.patch_embedding(x)
       #print("After patches: ", x.shape)
       
       x = self.NEW_PURE_Transformer3d(x)
       
       #print("After ViT: ", x.shape)
       
       x = self.decoder(x)
       #print("After decoder: ", x.shape)
       return x
