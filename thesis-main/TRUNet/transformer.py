import torch
import torch.nn as nn
from torch.nn import Dropout, Conv3d, LayerNorm
from torch.nn.modules.utils import _triple
from block import Block
import copy
import numpy as np
from double_conv_residual import SingleConv, DoubleConv


#This is for the hybrid models
class Transformer3d(nn.Module):
    def __init__(self, config):
        super(Transformer3d, self).__init__()
        self.config = config

        patch_size = 1 
        # 1x1x1 conv according to TRUNet paper
        self.patch_embeddings = Conv3d(in_channels = config.decoder_channels[0],
                                       out_channels = config.hidden_size, #Decides how many embeddings I have 
                                       kernel_size = 1, #According to TRUNet paper
                                       stride = 1) #Accorindg to TRUNet paper
        self.dropout = Dropout(config.transformer_dropout_rate)


        #------------------Encoder part from papers-----------------------
        self.layer = nn.ModuleList() #Anmar:Holds submodules in a list
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # make the layers within the transformer
        for _ in range(config.transformer_num_layers):
            # block contains multi-head attention and mlp
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer)) #Anmar: Appends to the module list. 
        
        #------------------from decoder_cup from papers--------------
      
        # Old
        #self.last_conv = nn.Sequential(
        #   nn.Conv3d(config.hidden_size, config.decoder_channels[0], kernel_size=3, stride=1, padding=1, dtype=torch.float32),
        #   nn.ReLU()  # Adding ReLU activation
        #)
        
        # Single conv with actvation function and normalization
        self.last_conv = DoubleConv(config.hidden_size, config.decoder_channels[0], config.activation, config.normalization)

        #TODO Try again...
        #self.last_conv = DoubleConv(config.hidden_size, config.decoder_channels[0], use_residual=False)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.n_patches**3, self.config.hidden_size))

    def forward(self, x):
    
        #x is the output from the CNN. 
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) 


        #Flatten the spatials dimensions after the CNN. 
        x = x.flatten(2) 

        x = x.transpose(-1, -2)  # (B, n_patches, hidden) #TODO Just like in the video, we transpose. So 2744, 768. 

        #print(self.position_embeddings.shape)

        hidden_states = x + self.position_embeddings 

        #------------------------------------------
        #In each block, attention is being done
        for layer_num, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states, layer_num) #Anmar: Forward() for block function
            
        hidden_states = self.encoder_norm(hidden_states) #One extra layer norm that's not needed??
    
        #quit()
        #Reshaping
        B, n_patch, hidden = hidden_states.size()
        h, w, l = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch)) #Takes square root of the patches
        x = hidden_states.permute(0, 2, 1) #Switches the 1,2744,768 to 1,768,2744, i.e. STILL a matrix
        x = x.contiguous().view(B, hidden, h,w,l) #TODO Investigate if contiguous  is needed?
        
        x = self.last_conv(x) 

        return x
        

        #output here is the output which will be used for the decoder. 

