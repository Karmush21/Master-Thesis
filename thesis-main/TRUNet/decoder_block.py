# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import *

from double_conv_residual import DoubleConv



#TODO CHANGE THESE LATER!
class DecoderBlock3d(nn.Module):
    def __init__(
            self, decoder_channels, num_classes, activation, normalization, use_residual):
        super(DecoderBlock3d, self).__init__() #Anmar: Calls the constructor of the nn.Module class.
        self.moduleList = nn.ModuleList()
        self.decoder_channels = decoder_channels
        for i in range(len(decoder_channels)-1):
            upconv = nn.ConvTranspose3d(
                in_channels=self.decoder_channels[i],
                out_channels=self.decoder_channels[i]//2,
                kernel_size=2,
                stride=2,
                padding=0,
                dtype=torch.float32
            )
            self.moduleList.append(upconv)
          
            conv = DoubleConv(
                self.decoder_channels[i], 
                self.decoder_channels[i+1],
                activation,
                normalization,
                use_residual=use_residual #TODO Use residual in decoder as well??
            )

            self.moduleList.append(conv)
        
        self.last_conv = nn.Conv3d(
                    in_channels=self.decoder_channels[-1],
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )

    def forward(self, x, features):

        #print(self.moduleList)
        for i in range(len(self.decoder_channels)-1):
            x = self.moduleList[i*2](x)
            #print("After UpConv: ", x.shape)
          
            
            #If the x and fetures[i] volumes just differ by 1. Happens with 1024 channels           
            #if features[i].shape[-1] - x.shape[-1] == 1:
            #    print("here?")
            #    x = F.pad(x, (1, 0, 1, 0, 1, 0))  # Pad 1 element at the beginning of each dimension
            
            x = torch.cat([x, features[i]], dim=1)
            #print("After cat: ", x.shape)
          


            #print("After Cat " +str(i) + ": ", x.shape)
            x = self.moduleList[i*2 + 1](x)
            
           
            #print("After Conv: ", x.shape) #Will eventually be the final conv to N classes.
            #print("\n")
        
        
        x = self.last_conv(x)
        
        
        x = F.softmax(x, dim=1)

        return x
    
if __name__ == "__main__":
    encoder_channels = [32,64,128,256,512,1024]
    decoder_channels = [256,128,64,32]
    x = torch.rand(1, 256, 14,14,14)
    features = [
        torch.rand(1, 32, 224, 224, 224),
        torch.rand(1, 64, 112, 112, 112),
        torch.rand(1, 128, 56, 56, 56),
        torch.rand(1, 256, 28, 28, 28),
    ]

    decoder = DecoderBlock3d(decoder_channels, 2, False)
    y = decoder(x, list(reversed(features)))
