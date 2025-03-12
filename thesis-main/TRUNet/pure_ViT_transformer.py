import torch
import torch.nn as nn
from torch.nn import Dropout, Conv3d, LayerNorm
from torch.nn.modules.utils import _triple
from block import Block
import copy
import numpy as np
from double_conv_residual import SingleConv, DoubleConv


#Transformer3d Class for the Pure ViT network.
class Pure_ViT_Transformer3d(nn.Module):
    def __init__(self, config):
        super(Pure_ViT_Transformer3d, self).__init__()
        self.config = config

        #------------------Encoder part from papers-----------------------
        self.layer = nn.ModuleList() #Anmar:Holds submodules in a list
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # make the layers within the transformer
        for _ in range(config.transformer_num_layers):
            # block contains multi-head attention and mlp
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer)) #Anmar: Appends to the module list. 

    def forward(self, hidden_states):
        #------------------------------------------
        #In each block, attention is being done
        for layer_num, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states, layer_num) #Anmar: Forward() for block function
            
        hidden_states = self.encoder_norm(hidden_states) #One extra layer norm that's not needed??
        quit()
        
        #TODO Move to decoder for now, trying without pointwise convolution
        #Reshaping
        B, n_patch, hidden = hidden_states.size()
        h, w, l = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch)) #Takes square root of the patches
        x = hidden_states.permute(0, 2, 1) #Switches the 1,2744,768 to 1,768,2744, i.e. STILL a matrix
        x = x.contiguous().view(B, hidden, h,w,l) #TODO Investigate if contiguous  is needed?
        

        
        #output here is the output which will be used for the decoder. 
        #x = hidden_states #TODO, remove later if you do reshape
        
        
        
        return x
