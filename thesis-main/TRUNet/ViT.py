# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage


from transformer import Transformer3d
from functions import *
from cnn import CNN
#from new_transformer import NEW_Transformer3d
from decoder_block import DecoderBlock3d

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class VisionTransformer3d(nn.Module): #Has to inhert from nn.module
    def __init__(self, config, use_transformer = True):
        #print("Initialize model in ViT.py")
        super().__init__()  # defining this as a model in nn
        
        self.config = config
        self.use_transformer = use_transformer
        #self.transformer = Transformer3d(config)
        #self.decoder = DecoderCup3d(config)


        #TODO Just send the whole config instead right? like in new transformer
        self.CNN_encoder = CNN(in_channels=config.in_channels, out_channels=config.encoder_channels, 
                               activation = config.activation,
                               normalization = config.normalization, use_residual=config.use_residual, 
                               use_stride_conv_downsampling = config.use_stride_conv_downsampling)
        
        if use_transformer:
            self.NEW_Transformer = Transformer3d(self.config)
        
        
        #TODO Just send the whole config instead right? like in new transformer
        self.Decoder = DecoderBlock3d(decoder_channels = config.decoder_channels, num_classes=config.n_classes, 
                                      activation = config.activation,
                                      normalization= config.normalization, use_residual=config.use_residual)





    def forward(self, x):
        # Old code where we copy the volume three times :)
        #if x.size()[1] == 1:
        #    x = x.repeat(1, 3, 1, 1, 1)  # turning the image into 3 channels. #Anmar: Basically copies the volumes 3 times I think...
        
        #Anmar: Zero pad volumes that they get shape 512,512,512. but perhaps we can change this if we do rescale or something like this?
        #x = convert_volume_to_correct_shape(x)


        #TODO: Do instead Encoder->Transformer->Decoder! Not the way things are being done now with transformer. 

        # first run transformer
        #x, features = self.transformer(x)  # (B, n_patch, hidden) #Anmar: features are for skip connections. We have 3 of them. 

        
        # then decoder
        #x = self.decoder(x, features) 
        
        #return x
        
        #TODO, This will later be the final idea, more clean I would say.
        x, featuers = self.CNN_encoder(x)
       
        if self.use_transformer:
            # Pass through transformer only if no_transformer is False
            x = self.NEW_Transformer(x)

        
        x = self.Decoder(x, featuers)
        return x
