import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import re
import matplotlib.pyplot as plt

def sort_filenames(filenames, volumes: bool):
    """
    Sort filenames based on patient (pt) and time (dt) numbers extracted from the filename.

    Args:
        filenames (list of str): List of filenames to be sorted.
        volumes (bool): Whether the filenames are for volumes (True) or masks (False).

    Returns:
        list of str: Sorted list of filenames.
    """
    def extract_numbers(filename):
        if volumes:
            match = re.search(r'ct_pt(\d+)_dt(\d+)', filename)  # Extract pt and dt numbers for volumes
        else:
            match = re.search(r'pt(\d+)_dt(\d+)', filename)  # Extract pt and dt numbers for masks
        if match:
            pt_num = int(match.group(1))
            dt_num = int(match.group(2))
            return pt_num, dt_num
        return float('inf'), float('inf')  # Return a high value if no match to sort unmatched files last

    return sorted(filenames, key=extract_numbers)

def filter_filenames(filenames, remove_volumes):
    """
    Filter out filenames corresponding to patients that should be removed.

    Args:
        filenames (list of str): List of filenames to be filtered.
        remove_volumes (list of str): List of patient numbers to remove.

    Returns:
        list of str: Filtered list of filenames.
    """
    def should_remove(filename):
        match = re.search(r'pt(\d+)', filename)  # Extract patient number from filename
        if match:
            pt_num = match.group(1)
            return pt_num in remove_volumes  # Check if patient is in the removal list
        return False

    return [f for f in filenames if not should_remove(f)]

def replace_with_changed_masks(mask_files, changed_mask_folder):
    """
    Replace original mask files with updated versions if available.

    Args:
        mask_files (list of str): List of original mask filenames.
        changed_mask_folder (str): Path to the folder containing changed masks.

    Returns:
        list of str: List of mask filenames, with changed masks replacing originals where applicable.
    """
    # List all changed masks in the folder, filtering by specific pattern and file extensions
    changed_mask_files = [f for f in os.listdir(changed_mask_folder) if '_AK' in f and (f.endswith('.nii') or f.endswith('.nii.gz'))]
    changed_mask_files_dict = {re.sub(r'_AK', '', re.sub(r'(\.nii\.gz|\.nii)$', '', f)): f for f in changed_mask_files}

    replaced_mask_files = []
    for f in mask_files:
        base_filename = re.sub(r'(\.nii\.gz|\.nii)$', '', f)  # Remove extensions to get the base name
        if base_filename in changed_mask_files_dict:
            # If a changed mask exists, replace the original mask filename
            replaced_mask_files.append(changed_mask_files_dict[base_filename])
        else:
            replaced_mask_files.append(f)
    
    return replaced_mask_files

def plot_slices(volume_data, mask_data, num_slices=5, alpha=0.7):
    """
    Plot slices of the volume and corresponding mask data for sanity checking.

    Args:
        volume_data (numpy array): 3D array of volume data.
        mask_data (numpy array): 3D array of mask data.
        num_slices (int): Number of slices to plot.
        alpha (float): Transparency level for overlaying mask data on volume slices.
    """
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 6))
    slice_indices = np.linspace(150, 200, num_slices, dtype=int)  # Define the slice indices to plot
    
    for i, idx in enumerate(slice_indices):
        axes[i].imshow(volume_data[:, :, idx], cmap='gray')  # Plot volume slice
        axes[i].imshow(mask_data[:, :, idx], cmap='jet', alpha=alpha)  # Overlay mask slice
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')  # Hide axis
    
    plt.tight_layout()
    plt.show()

# Define the desired output size for downsampling
output_size = (224, 224, 224)

######### DEFINING PATHS #########
# Define paths to the folders containing volume, mask, and changed mask data
volume_folder = '/mnt/anmka/fenix/cdrg/TimeResolvedCT/ct_as_nifti_new'
mask_folder = '/mnt/anmka/fenix/cdrg/TimeResolvedCT/myo_280524'
changed_mask_folder = '/mnt/anmka/fenix/cdrg/TimeResolvedCT/myo_280524_anmar_changes'
output_folder = '/mnt/anmka/fenix/cdrg/TimeResolvedCT/myo_280524_npz'

# Ensure the output folder exists or create it if it doesn't
os.makedirs(output_folder, exist_ok=True)
##################################

######### READING IN VOLUMES AND MASKS AND SORTING THEM BASED ON PATIENT NUMBER #########
# List and sort volume and mask files based on patient and time numbers
volume_files = sorted([f for f in os.listdir(volume_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])

# Use different patterns for sorting volumes and masks
volume_files = sort_filenames(volume_files, volumes=True)
mask_files = sort_filenames(mask_files, volumes=False)
########################################################################################

print(f'Original number of volumes and their corresponding masks: {len(volume_files)} | {len(mask_files)}')

######### REMOVING VOLUMES THAT WE DON'T WANT TO PROCESS RIGHT NOW #########
# Define lists of patient numbers to remove or review
remove_volumes = ["8", "11", "48", "59"]  # Patients that definitely need to be excluded
doubtful_volumes = ["20", "26", "30", "33", "39", "52", "58"]  # Patients to review before processing

# TODO: Assign patient 2 to the test set and patient 58 to the validation set

# Filter out volumes and masks based on the removal list
volume_files = filter_filenames(volume_files, remove_volumes)
mask_files = filter_filenames(mask_files, remove_volumes)
###############################################################################

print(f'Number of volumes and masks after removal: {len(volume_files)} | {len(mask_files)}')

######### REPLACING MASKS WITH CHANGED VERSIONS IF APPLICABLE #################
# Decide whether to use changed masks or original masks
use_changed_masks = False

if use_changed_masks:
    mask_files = replace_with_changed_masks(mask_files, changed_mask_folder)
###############################################################################

# Ensure that the number of volume files matches the number of mask files
if len(volume_files) != len(mask_files):
    raise ValueError("The number of volume files does not match the number of mask files")

# Process each volume and mask file pair
index = 0
for volume_file, mask_file in zip(volume_files, mask_files):
    # Create a filename for the output .npz file by removing both .nii and .gz extensions
    base_filename = volume_file
    if base_filename.endswith('.nii.gz'):
        base_filename = base_filename[:-7]  # Remove the '.nii.gz' extension
    elif base_filename.endswith('.nii'):
        base_filename = base_filename[:-4]  # Remove the '.nii' extension
    
    output_filename = base_filename + '.npz'
    output_path = os.path.join(output_folder, output_filename)
    
    # Skip processing if the .npz file already exists
    if os.path.exists(output_path):
        print(f"The file: {output_path} already exists. Skipping...")
        continue

    print(f'{volume_file} | {mask_file}')
    
    # Construct the full paths to the volume and mask files
    volume_path = os.path.join(volume_folder, volume_file)
    mask_path = os.path.join(mask_folder, mask_file) if not use_changed_masks else os.path.join(changed_mask_folder, mask_file)

    # Load the NIfTI files
    volume_nifti = nib.load(volume_path)
    mask_nifti = nib.load(mask_path)

    # Extract data arrays from the NIfTI objects
    volume_data = volume_nifti.get_fdata()
    mask_data = mask_nifti.get_fdata()

    # Optional sanity check by plotting slices (currently commented out)
    #plot_slices(volume_data, mask_data)

    # Downsample the volume and mask data to the desired output size
    x, y, z = volume_data.shape
    volume_data = zoom(volume_data, (output_size[0] / x, output_size[1] / y, output_size[2] / z), order=3)
    mask_data = zoom(mask_data, (output_size[0] / x, output_size[1] / y, output_size[2] / z), order=0)

    # Optional sanity check after downsampling (currently commented out)
    #plot_slices(volume_data, mask_data)

    # Save the volume and mask data as a compressed .npz file
    np.savez_compressed(output_path, image=volume_data, label=mask_data)

    print(f"Saved {output_path}")
    index += 1

print("All files processed and saved.")
