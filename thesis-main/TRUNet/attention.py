# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Linear
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm #For progress bar.


# Define the custom colormap from black to red
colors = [(0, 0, 0), (1, 0, 0)]  # Black to Red
n_bins = 100  # Number of bins #Prev 100
cmap_name = 'black_red'
black_red_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

save_attention_maps_bool = True #TODO, make this a parameter in the function instead. 

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_attention_maps(attention_probs, layer_num, output_dir='attention_maps', crop_region=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Assuming attention_probs has shape [1, num_heads, 2744, 2744]
    attention_probs = attention_probs[0].detach().cpu().numpy()  # Shape [num_heads, 2744, 2744]

    num_heads = attention_probs.shape[0]

    for head in tqdm(range(num_heads)):
        # Extract the attention map for the current head
        attention_map = attention_probs[head]
        
        # Extract the a of the attention map
        first_row = attention_map[0, :]
        
        # Reshape the first row to be a 2D array with a single row
        first_row_2d = np.expand_dims(first_row, axis=0)  # Shape [1, 2744]

        #print(first_row_2d)
        
        
        # Normalize the first row if needed for better visibility
        #first_row_normalized = (first_row_2d - np.min(first_row_2d)) / (np.max(first_row_2d) - np.min(first_row_2d))
        
        #print(first_row_normalized.shape)
        first_row_2d = first_row_2d.tolist()[0]


        #print(first_row_normalized)
        #quit()


        # Save the first row as an SVG image
        fig_width = 20  # Width in inches
        fig_height = 5  # Height in inches (adjusted to fit tick labels better)
        plt.figure(figsize=(fig_width, fig_height), dpi=500)  # Set figure size and DPI
        
        
        #plt.imshow(first_row_normalized, cmap=black_red_cmap, aspect=500)
        #plt.axis('on')  # Show axes
        #plt.title(f'Layer {layer_num+1} - Head {head+1} | Last Row')
        
        
        plt.plot(first_row_2d, color='red')  # Use markers for individual points
        plt.grid(True)
        plt.xlabel('Patch Index')
        plt.ylabel('Normalized Attention Score')
        plt.title(f'Layer {layer_num} - Head {head} - Row Attention Scores')
        
        # Set x-axis tick marks and labels
        num_ticks = 10
        tick_positions = np.linspace(0, 2743, num=num_ticks)
        tick_labels = np.linspace(0, 2743, num=num_ticks).astype(int)
        plt.xticks(ticks=tick_positions, labels=tick_labels)
        
        y_min, y_max = min(first_row_2d), max(first_row_2d)
        num_y_ticks = 10
        plt.yticks(np.linspace(y_min, y_max, num=num_y_ticks))
       
        # Save the figure as SVG
        plt.savefig(os.path.join(output_dir, f'layer{layer_num+1}_head{head+1}.svg'), bbox_inches='tight', pad_inches=0)
        plt.close()





class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.num_attention_heads = config.transformer_num_heads
        
        # For multi-head attention, we make the head dimension smaller.
        # So total computonal cost is similar to single-head attention
        
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        
        #12*64. Should match hidden size
        self.all_head_size = self.num_attention_heads * self.attention_head_size 
        
        #print("How much each head sees from the embedding representation: ", self.attention_head_size)
        
        
        #Creates the three matricies that we will use with the embedding. All of these are 768x768. So equal weights for all.
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

    
        
        self.out = Linear(config.hidden_size, config.hidden_size) #Anmar: The last linear layer, to get back to original size. After concat
        self.attn_dropout = Dropout(config.transformer_attention_dropout_rate)
        self.proj_dropout = Dropout(config.transformer_attention_dropout_rate)

        self.softmax = Softmax(dim=-1) #TODO Check if this is correct

    
    #Anmar: I don't understand the purpose of this function. 
    def transpose_for_scores(self, x): 
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)



    def forward(self, hidden_states, layer_num):
        
        #Anmar: Embeddings through linear projections. All shapes are 2744x768.
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states) 


        #Anmar: Pretty sure we divide so each head only sees of the attention filter.
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #Anmar: This is a square matrix: 2744x2744. 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        #save_attention_maps(attention_scores, layer_num, f"attention_maps/plot/")
        #quit()
        
        attention_probs = self.softmax(attention_scores) #[B, 12, 2744, 2744]. I.e. the attention map for each head.


        #save_attention_maps(attention_probs, layer_num, f"attention_maps/plot_after_softmax/")
        #quit()


        attention_probs = self.attn_dropout(attention_probs)

     

        
        context_layer = torch.matmul(attention_probs, value_layer)

        
        
        #Somewhere here is where the concatenation is being done. Before the final linear layer.
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        

        context_layer = context_layer.view(*new_context_layer_shape)
        


        

        attention_output = self.out(context_layer)
        
        
        attention_output = self.proj_dropout(attention_output)

        
        return attention_output 
