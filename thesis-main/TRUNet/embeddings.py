# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Dropout, Conv3d
from torch.nn.modules.utils import _triple

from cnn import CNN


class Embeddings3d(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config):
        super(Embeddings3d, self).__init__()
        #self.hybrid = None
        self.config = config
        # img_size = _triple(img_size)

        # #TODO This could be wrong
        # if config.patches.get("grid") is not None:  # ResNet
        #     grid_size = config.patches["grid"]
        #     patch_size = (
        #     img_size[0] // config.patches.size // grid_size[0], img_size[1] // config.patches.size // grid_size[1],
        #     img_size[2] // config.patches.size // grid_size[2])
        #     patch_size_real = (patch_size[0] * config.patches.size, patch_size[1] * config.patches.size,
        #                        patch_size[2] * config.patches.size)
        #     n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (
        #                 img_size[2] // patch_size_real[2])
            
        #     self.hybrid = True
        # else:
        #     patch_size = _triple(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        #     self.hybrid = False

        #if self.hybrid:
            #self.hybrid_model = ResNetV23d(block_units=config.resnet.num_layers,
            #                               width_factor=config.resnet.width_factor)
            #in_channels = self.hybrid_model.width * 16

        self.CNN_encoder = CNN(in_channels=config.in_channels, out_channels=config.encoder_channels, use_residual=config.use_residual)
        # else:
        #     self.convolutions = EncoderCup3d(config)
        #     if len(config.decoder_channels) == 4:
        #         in_channels = config.decoder_channels[-4]
        #     elif len(config.decoder_channels) == 3:
        #         in_channels = config.decoder_channels[-3]
        
        
        #TODO
        #Anmar: This convolution is the linear projection, I guess it can be seen as convolution right?
        #Anmar: patch_size is 1x1x1 which does make sense because we just want to extract patches. i.e. not do any e.g. adding.
        #This is what TransUnet wants. However could be wrong and might and to change it.
        patch_size = 1
        self.patch_embeddings = Conv3d(in_channels=config.encoder_channels[-1],
                                       out_channels=config.hidden_size, #Decides how many embeddings I have 
                                       kernel_size=patch_size,
                                       stride=patch_size)

        #Anmar: For now we hardcode it, but change this later
        n_patches = 14**3

        #Should be fined that these are like this, they get learnet during training
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # self.position_embeddings = nn.Parameter(torch.reshape(torch.arange(0,n_patches * config.hidden_size), (1, n_patches, config.hidden_size)).float())

        self.dropout = Dropout(config.transformer_dropout_rate)

    def forward(self, x):
        #print("Forward method Embedding class")
        #print("Start with shape: ", x.shape, x.dtype)

        x, features = self.CNN_encoder(x)
        
        #Hidden feature in Lee's paper
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) #Anmar: 768, 14,14,14 #I wonder if this is needed now??
        #n_patches = x.shape[-1] ** 3

        #print("After patching: ", x.shape)

        #Flattening is the linear projection! We could add the a linear layer here bot not needed. 
        x = x.flatten(2) 
        
        x = x.transpose(-1, -2)  # (B, n_patches, hidden) #Anmar: Just like in the video, we transpose
        

        #Adding Positional embeddings
        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.config.hidden_size, device=x.device))

        embeddings = x + self.position_embeddings 
        
        #If we want dropout. 
        #embeddings = self.dropout(embeddings)

        #Anmar: embedings shape is (2744, 768) Meaning that we have 2744 embeddings and each embedding contains 768 elements. 
        return embeddings, features
