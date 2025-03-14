# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import Dropout, Linear


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer_mlp_dim)
        self.fc2 = Linear(config.transformer_mlp_dim, config.hidden_size)
        
        #self.act_fn = nn.functional.gelu
        #self.act_fn = nn.ReLU()  
        self.act_fn = self.get_activation(config.activation)
        
        self.dropout = Dropout(config.transformer_dropout_rate)

        self._init_weights()
    
    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f'Activation type "{activation}" not supported.')


    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        #print("Forward pass MLP")
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
