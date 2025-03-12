import torch
import torch.nn as nn
import torch.nn.functional as F
from double_conv_residual import DoubleConv

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, activation, normalization, use_residual, use_stride_conv_downsampling):
        super(CNN, self).__init__()

        self.num_layers = len(out_channels)
        self.features = nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                conv_layer = DoubleConv(in_channels, out_channels[i], activation, normalization, use_residual)
            else:
                conv_layer = DoubleConv(out_channels[i-1], out_channels[i], activation, normalization, use_residual)
            self.features.append(conv_layer)
            
            # Add down-sampling layer only if it's not the last layer
            if i < self.num_layers - 1:
                # Decide how we want to downsample. 3D unet paper from nnunet guys do strideconv downsampling.
                if use_stride_conv_downsampling:
                    down_sampling_block = nn.Sequential(
                        nn.Conv3d(out_channels[i], out_channels[i], kernel_size=2, stride=2), #input and output can be equal
                        self.get_normalization(normalization, out_channels[i]),
                        self.get_activation(activation) 
                    )
                    self.features.append(down_sampling_block)  # Strided convolution for downsampling
                else:         
                    self.features.append(nn.MaxPool3d(kernel_size=2, stride=2))
                
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
        features_list = []
        for idx, layer in enumerate(self.features):
            #print(f'shape before {x.shape}')
            x = layer(x)
            #print(f'shape After {x.shape}')
            if isinstance(layer, DoubleConv):
                features_list.append(x)
            #print("\n")
        
        
        features_list = features_list[:-1]
        features_list.reverse()
        
        return x, features_list
        
if __name__ == "__main__":
    x = torch.rand(1, 1, 224, 224, 224)
    model = CNN(in_channels=1, out_channels=[32, 64, 128, 256], activation='relu', normalization='instance', use_residual=False, use_stride_conv_downsampling=False)
    output, features = model(x)
0