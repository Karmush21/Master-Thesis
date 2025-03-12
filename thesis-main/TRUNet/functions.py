import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
import nibabel as nib
import os
import shutil
import subprocess
import numpy as np
import random
from scipy.ndimage import label

from monai.metrics import get_mask_edges, get_surface_distance
from glob import glob
import re
import os

def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, scheduler, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']

    print("Checkpoint loaded successfully.")
    #print(f"Model state dict keys: {list(model.state_dict().keys())}")
    #print(f"Optimizer state dict keys: {list(optimizer.state_dict().keys())}")
    #print(f"Scheduler state dict keys: {list(scheduler.state_dict().keys())}")
    #print(f"Starting from epoch: {epoch}")

    #scheduler_state_dict = scheduler.state_dict()
    #for key, value in scheduler_state_dict.items():
    #    print(f"{key}: {value}")
    
    return epoch

def load_model(model, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])


def retain_largest_component(tensor):
    print(tensor.shape)
    pred_np = tensor.squeeze().cpu().numpy()
    pred_np = np.argmax(pred_np, axis=0)  # (D, H, W)  
    print(pred_np.shape)
    l, nl = label(pred_np)
    sizes_l = [len(np.where(l==n+1)[0]) for n in range(nl)]
    keeper = np.where(sizes_l==np.nanmax(sizes_l))[0][0]+1 #finds the largets component
    pred_np = np.where(l==keeper,1,0)
    print(pred_np.shape, np.unique(pred_np))
    
    #l, nl = label(myo_new_smoothed)


def dice_score(pred, gt):  # data in shape [batch, classes, h, w, d], where gt is one-hot encoded
    dice = []

    # Loop over each batch_item
    for batchloop in range(gt.shape[0]):
        dice_tmp = []
        # Assuming classes include background as roi 0, single class without background should be handled
        # Loops over each channel for a particular volume in the batch
        for roi in range(gt.shape[1]):
            # Skip the background if there are multiple classes
            if gt.shape[1] > 1 and roi == 0:
                continue
            
            # Selects the correct volumes in the correct batch to do the prediction. 
            pred_tmp = pred[batchloop, roi]
            gt_tmp = gt[batchloop, roi]
            
            # a checks where the two overlap. b,c just adds the total set. 
            a = torch.sum(pred_tmp[gt_tmp == 1])
            b = torch.sum(pred_tmp)
            c = torch.sum(gt_tmp)
            
            #Deal with division by zero
            if b + c == 0:
                metric = 0.0
            else:
                metric = a * 2.0 / (b + c)
            # Append for the current class and then continue with a new class.
            dice_tmp.append(metric)
        
        if dice_tmp:
            dice.append(torch.mean(torch.tensor(dice_tmp, device=pred.device)))
        else:
            dice.append(torch.tensor(0.0, device=pred.device))
    return torch.mean(torch.tensor(dice, device=pred.device))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def polynomial_decay(epoch, gamma=0.9, total_epochs=200):
    return (1 - (epoch / total_epochs)) ** gamma

def pad_data(data, target_x=300, target_y=300, target_slices=0):
    if data.shape[0] != target_x or data.shape[1] != target_y or data.shape[2] != target_slices:
        pad_x = max(0, target_x - data.shape[0])
        pad_y = max(0, target_y - data.shape[1])
        pad_z = max(0, target_slices - data.shape[2])
        pad_width = ((0, pad_x), (0, pad_y), (0, pad_z))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    return data


def calculate_class_distribution(label):
    '''
    Calculcate the distribution of classes in a binary label array.
    ''' 
    #Remove batch dim
    label = label.squeeze()
    
    #Make into np if it's a torch
    label = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
    
    num_zeros = np.sum(label == 0)
    num_ones = np.sum(label == 1)
    
    total_elements = label.size
    percent_zeros = (num_zeros / total_elements) * 100
    percent_ones = (num_ones / total_elements) * 100
    
    return {
        'num_zeros': num_zeros,
        'num_ones': num_ones,
        'percent_zeros': percent_zeros,
        'percent_ones': percent_ones
    }

def unique_labels_tensor(volume_tensor):
    return np.unique(volume_tensor.detach().cpu().numpy())

def grad_parameters(model):
     for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, param.grad.abs().mean())


def save_as_nifti(output, filename, info_, use_info = False):
    if isinstance(output, torch.Tensor):
        print("never here no?")
        output = output.permute(1, 2, 0).cpu().numpy()
    
    #else: 
      #print(output.shape)
      #output = np.transpose(output, (1, 0, 2))
      #print(output.shape)
    if use_info:

       info_img = nib.load(info_) #TODO Add back when we do inference
        
       img = nib.Nifti1Image(output, info_img.affine, info_img.header)

    else:
        img = nib.Nifti1Image(output, np.eye(4))

    # Save the NIfTI image
    nib.save(img, filename)

def delete_folder_and_contents(root_folder, target_folder_name):
    # Iterate through all the directories and files in the root folder
    for root, dirs, files in os.walk(root_folder):
        # Check if the target folder name is in the list of directories
        if target_folder_name in dirs:
            target_folder_path = os.path.join(root, target_folder_name)
            # Delete the contents inside the target folder
            for item in os.listdir(target_folder_path):
                item_path = os.path.join(target_folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                else:
                    shutil.rmtree(item_path)  # Recursively delete subdirectories
            # Delete the target folder itself
            os.rmdir(target_folder_path)

def print_nvidia_smi():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))


def generate_encoder_channels(start, length, monai_unet):
    channels = [start]
    for _ in range(length - 1):
        channels.append(channels[-1] * 2)
    
    # Because monai u-net wants the start channels at the end of the tuple ...
    if monai_unet:
        channels.append(start)
    
    return tuple(channels)

#Helper function for list_pth_files_in_folder function
def extract_epoch(file_name):
    """
    Extract the epoch number from a file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        int: The epoch number extracted from the file name.
    """
    match = re.search(r'epoch_(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

def list_pth_files_in_folder(folder_path):
    """
    List and sort all .pth files in the specified folder based on epoch number.

    Args:
        folder_path (str): The path to the folder containing .pth files.

    Returns:
        List[str]: A sorted list of .pth file names based on epoch numbers.
    """
    # Ensure the folder path is valid
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path {folder_path} is not a directory or does not exist.")
    
    # Find all .pth files in the folder
    pth_files = glob(os.path.join(folder_path, "*.pth"))

    # Sort the list of .pth files based on epoch number
    sorted_files = sorted(
        pth_files,
        key=lambda f: extract_epoch(os.path.basename(f))
    )
    
    # Return only the file names (not the full paths)
    return [os.path.basename(f) for f in sorted_files]




def select_evenly_spaced_epochs(file_paths, num_samples):
    """
    Selects `num_samples` evenly spaced epochs from a list of file paths.

    Parameters:
    - file_paths: List of file paths (e.g., ["best_metric_model_epoch_10.pth", "best_metric_model_epoch_20.pth", ...])
    - num_samples: Number of epochs to select

    Returns:
    - A list of selected file paths
    """
    # Extract epoch numbers from file names
    epoch_pattern = re.compile(r'best_metric_model_epoch_(\d+)\.pth')
    epochs = []
    
    for file in file_paths:
        match = epoch_pattern.search(file)
        if match:
            epoch_number = int(match.group(1))
            epochs.append(epoch_number)
    
    # Sort epoch numbers
    epochs = sorted(set(epochs))
    
    if not epochs:
        raise ValueError("No valid epoch numbers found in file names.")
    
    # Calculate step size
    num_epochs = len(epochs)
    step = (num_epochs - 1) / (num_samples - 1)

    # Select evenly spaced epochs
    selected_epochs = [epochs[int(round(i * step))] for i in range(num_samples)]
    
    # Get the corresponding file paths
    selected_files = [file for file in file_paths if any(f'best_metric_model_epoch_{epoch}.pth' in file for epoch in selected_epochs)]
    
    return selected_files

# # Example usage
# file_paths = [
#     "best_metric_model_epoch_10.pth",
#     "best_metric_model_epoch_20.pth",
#     "best_metric_model_epoch_45.pth",
#     "best_metric_model_epoch_11.pth",
#     "best_metric_model_epoch_219.pth",
#     "best_metric_model_epoch_59.pth",
#     "best_metric_model_epoch_154.pth",
#     "best_metric_model_epoch_29.pth",
#     "best_metric_model_epoch_5.pth",
#     "best_metric_model_epoch_15.pth",
#     "best_metric_model_epoch_2.pth",
#     "best_metric_model_epoch_68.pth",
#     "best_metric_model_epoch_16.pth",
#     "best_metric_model_epoch_322.pth",
#     "best_metric_model_epoch_69.pth",
#     "best_metric_model_epoch_185.pth",
#     "best_metric_model_epoch_3.pth",
#     "best_metric_model_epoch_8.pth",
#     "best_metric_model_epoch_18.pth",
#     "best_metric_model_epoch_41.pth",
#     "best_metric_model_epoch_95.pth"
# ]

# selected_files = select_evenly_spaced_epochs(file_paths, num_samples=5)
# print("Selected files:", selected_files)
