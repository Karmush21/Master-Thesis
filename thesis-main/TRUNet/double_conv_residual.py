import torch
import torch.nn as nn
from functions import *


#TODO Change name since it has both doubleconv and singleconv class!
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, activation, normalization, use_residual=False):
        super(DoubleConv, self).__init__()
        self.use_residual = use_residual
        
        self.activation = self.get_activation(activation)
        
        if not use_residual:
            self.conv = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1, dtype=torch.float32),
                self.get_normalization(normalization, out_c),
                self.activation,
                nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, dtype=torch.float32),
                self.get_normalization(normalization, out_c),
                self.activation,
            )
        
        else:
            # This type of residual seem to a newer version compared to the old way
            # where addition is done before final activation
            # Paper: Identity Mappings in Deep Residual Networks, kaiming he et. al
            self.conv = nn.Sequential(
                self.get_normalization(normalization, in_c),
                self.activation,
                nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1, dtype=torch.float32),
                self.get_normalization(normalization, out_c),
                self.activation,
                nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, dtype=torch.float32),
                
            )
            
            
            self.residual = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0, dtype=torch.float32),
                #self.get_normalization(normalization, out_c) #TODO Investigate if this is needed
            )

    def get_normalization(self, normalization, out_c):
        if normalization == 'instance':
            return nn.InstanceNorm3d(out_c)
        
        elif normalization == 'batch':
            return nn.BatchNorm3d(out_c)
        
        elif normalization == 'layer':
            # Use GroupNorm with 1 group (equivalent to LayerNorm)
            return nn.GroupNorm(1, out_c)
        
        else:
            raise ValueError(f'Normalization type "{normalization}" not supported.')


    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f'Activation type "{activation}" not supported.')

                
        

    def forward(self, x):
        if not self.use_residual:
            out = self.conv(x)
            return out
        
        #Residual connections
        #print("use residual: ", x.shape)
        residual = self.residual(x)
        out = self.conv(x)
        
        out = torch.add(out, residual) #Can't do +=. You'll get issues with backpropagation. 
        #out = self.activation(out) #TODO I don't think you want this one. 
        return out

class SingleConv(nn.Module):
    def __init__(self, in_c, out_c, activation, normalization):
        super(SingleConv, self).__init__()
        self.activation = self.get_activation(activation)
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1, dtype=torch.float32),
            self.get_normalization(normalization, out_c),
            self.activation,
        )
    
    def get_normalization(self, normalization, out_c):
        if normalization == 'instance':
            return nn.InstanceNorm3d(out_c)
        elif normalization == 'batch':
            return nn.BatchNorm3d(out_c)
        elif normalization == 'layer':
            return nn.GroupNorm(1, out_c)
        else:
            raise ValueError(f'Normalization type "{normalization}" not supported.')

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f'Activation type "{activation}" not supported.')

    def forward(self, x):
        out = self.conv(x)
        return out
