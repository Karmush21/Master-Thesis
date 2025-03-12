import torch
import torch.nn as nn

# Define functions to calculate MSE, MAE, and Cosine Similarity
def mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()

def mae(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2)).item()

def cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    return F.cosine_similarity(tensor1_flat, tensor2_flat, dim=0).item()


# Set a random seed for reproducibility
torch.manual_seed(42)

# Create two random tensors
tensor1 = torch.randn(1, 5, 5, 4)  # Shape (batch size, channels, depth, height, width)
tensor2 = tensor1.permute(0, 3, 1, 2)  # Shape (batch size, channels, height, width, depth)



# Define a 3D convolutional layer with no bias
conv3d = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

# Apply the same kernel to both tensors
output1 = conv3d(tensor1)
output2 = conv3d(tensor2)



print("Output shape for tensor1:", output1.shape)
print("Output shape for tensor2:", output2.shape)
print("Output for tensor1:\n", output1)
print("Output for tensor2:\n", output2)


# mse_value = mse(output1, output2)
# mae_value = mae(output1, output2)
# cosine_sim_value = cosine_similarity(output1, output2)
