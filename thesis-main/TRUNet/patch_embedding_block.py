# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import matplotlib.pyplot as plt

from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import deprecated_arg, ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_PATCH_EMBEDDING_TYPES = {"conv", "perceptron"}
SUPPORTED_POS_EMBEDDING_TYPES = {"none", "learnable", "sincos"}


device = torch.device("cpu") #cuda: gpu 0, cuda:1 gpu 1

#Splits volume into patches and applies the linear layer, as in the ViT paper. From UNETR code.
class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4,
        >>>                     proj_type="conv", pos_embed_type="sincos")

    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        pos_embed: str = "perceptron",
        proj_type: str = "perceptron",
        pos_embed_type: str = "learnable",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            proj_type: patch embedding layer type.
            pos_embed_type: position embedding layer type.
            dropout_rate: fraction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.proj_type = look_up_option(proj_type, SUPPORTED_PATCH_EMBEDDING_TYPES)
        self.pos_embed_type = look_up_option(pos_embed_type, SUPPORTED_POS_EMBEDDING_TYPES)

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.proj_type == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
        self.patch_dim = int(in_channels * np.prod(patch_size))

        self.patch_embeddings: nn.Module
        if self.proj_type == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.proj_type == "perceptron":

            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}


            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )

            #Put it on the gpu.
            #self.patch_embeddings = self.patch_embeddings.to(device)

            
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

        if self.pos_embed_type == "none":
            pass
        elif self.pos_embed_type == "learnable":
            trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif self.pos_embed_type == "sincos":
            grid_size = []
            for in_size, pa_size in zip(img_size, patch_size):
                grid_size.append(in_size // pa_size)

            self.position_embeddings = build_sincos_position_embedding(grid_size, hidden_size, spatial_dims)
        else:
            raise ValueError(f"pos_embed_type {self.pos_embed_type} not supported.")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        
        #A Way to see that we are actually extracting true patches from the volume. 
        #x = self.patch_embeddings[0](x)  # This is the Rearrange layer
        #print(x.shape)
        #Get back the patches. 
        #x_reshaped = x.view(1, 2744, *(16,16,16))
        #print(x_reshaped.shape)
        #patches = x_reshaped[0][0]
        #print(patches.shape)
        
        #slice_index = 9
        #slice_to_save = patches[:, :, slice_index ]  # Shape will be (16, 16)
        #print(torch.unique(slice_to_save))

        #file_name = f'slice_{slice_index + 1}.png'
        #plt.imsave(file_name, slice_to_save, cmap='gray')
        #print(f'Slice saved to {file_name}')
        #TODO Convert to np and plot?
        
        
        
      
        #Matrix, row says how many patches we have, colums, how many elements in each patch representation. 
        x = self.patch_embeddings(x)

        if self.proj_type == "conv":
            x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
