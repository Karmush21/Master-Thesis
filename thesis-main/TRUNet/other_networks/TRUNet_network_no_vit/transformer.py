# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from other_networks.TRUNet_network_no_vit.embeddings import Embeddings3d
from other_networks.TRUNet_network_no_vit.encoder import Encoder


class Transformer3d(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer3d, self).__init__()
        
        
        self.embeddings = Embeddings3d(config, img_size=img_size)
        #self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        #encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return embedding_output, features
