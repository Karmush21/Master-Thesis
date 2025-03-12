# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class pure_vit_decoder(nn.Module):
    def __init__(self, config):
        super().__init__()  
        self.config = config
        
        # self.pointwise_conv = nn.Conv3d(
        #                 in_channels = self.config.hidden_size,
        #                 out_channels = self.config.n_classes,
        #                 kernel_size = 3,
        #                 stride = 1,
        #                 padding=1,
        #                 dtype=torch.float32
        #             )
        
        self.pointwise_conv = nn.Conv3d(
                        in_channels = self.config.hidden_size,
                        out_channels = self.config.n_classes,
                        kernel_size = 1,
                        stride = 1,
                        padding=0,
                        dtype=torch.float32
                    )

       
        #self.linear = nn.Linear(self.config.hidden_size, 2)  # Project 768 channels to 2 channels

    def forward(self, x):
        #print(x.device)
        x = F.interpolate(x, size=224, mode='trilinear', align_corners=True)
        #print("Output from ViT ", x.shape)
        
        #TODO This idea did not seem to work that well.
        # x = self.linear(x)
        # #print("After Linear layer ", x.shape)
        # B, n_patch, hidden = x.size()
        # h, w, l = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch)) #Takes square root of the patches
        # x = x.permute(0, 2, 1) #Switches the 1,2744,768 to 1,768,2744, i.e. STILL a matrix
        # x = x.contiguous().view(B, hidden, h,w,l) #TODO Investigate if contiguous  is needed?
        # #print("Reshaping ", x.shape)
        # x = F.interpolate(x, size=224, mode='trilinear', align_corners=True)
        # #print("Trilinear interpolation " , x.shape)
        
        
        x = self.pointwise_conv(x)
        
        #print("After conv: ", x.shape)
        x = F.softmax(x, dim=1)
 
        return x


