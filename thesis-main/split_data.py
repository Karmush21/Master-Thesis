import os
import shutil
from sklearn.model_selection import train_test_split
import random
import numpy as np

'''
Split data in train val and test.
'''

def move_specfic_files(base_dir, patient_dicts, folder):
    if folder not in ['train', 'validation', 'test']:
        raise ValueError("Folder must be one of 'train', 'validation', or 'test'")
    
    for timepoints in patient_dicts.values():
        for timepoint in timepoints:
            source_path = os.path.join(base_dir, timepoint)
            destination_path = os.path.join(base_dir, folder, timepoint)
            shutil.move(source_path, destination_path)
                


def move_specfic_patients(base_dir, specified_patients):
  
    # Assuming all your .npz files are in this folder
    npz_files = [file for file in os.listdir(base_dir) if file.endswith('.npz')]
    npz_files = sorted(npz_files, key=lambda x: int(x.split('_')[1][2:]))
    


    # Creating folders for train, test, and validation sets if they don't exist
    folders = ['train', 'test', 'validation']
    for folder in folders:
       folder_path = os.path.join(base_dir, folder)
       if not os.path.exists(folder_path):
           os.makedirs(folder_path)

    
    patient_timepoints_dict_train = {}
    patient_timepoints_dict_validation = {}
    patient_timepoints_dict_test = {}

    for patient_number in specified_patients["train"]:
        patient_timepoints_dict_train[patient_number] = [file for file in npz_files if file.startswith(f'ct_{patient_number}_dt')]

    for patient_number in specified_patients["validation"]:
        patient_timepoints_dict_validation[patient_number] = [file for file in npz_files if file.startswith(f'ct_{patient_number}_dt')]
    
    
    for patient_number in specified_patients["test"]:
        patient_timepoints_dict_test[patient_number] = [file for file in npz_files if file.startswith(f'ct_{patient_number}_dt')]

    
    
    move_specfic_files(base_dir, patient_timepoints_dict_train, folder="train")
    move_specfic_files(base_dir, patient_timepoints_dict_validation, folder="validation")
    move_specfic_files(base_dir, patient_timepoints_dict_test, folder="test")
    
    
    print("Moved specific patients to a certain folder!")


def split_data(npz_dir, train_ratio, val_ratio, test_ratio):
    # Create directories for train, test, and validation sets
    train_dir = os.path.join(npz_dir, 'train')
    test_dir = os.path.join(npz_dir, 'test')
    val_dir = os.path.join(npz_dir, 'validation')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of npz files
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    # Group files by patient ID
    patient_files = {}
    for npz_file in npz_files:
        patient_id = npz_file.split('_')[1]  # Extract patient ID from file name
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(npz_file)
    
    # Shuffle the list of patient IDs
    patient_ids = list(patient_files.keys())
    np.random.shuffle(patient_ids)
    
    # Calculate number of patients for each split
    total_patients = len(patient_ids)
    train_count = int(total_patients * train_ratio / 100)
    test_count = int(total_patients * test_ratio / 100)
    val_count = total_patients - train_count - test_count
    
    # Function to move files
    def move_files(patient_list, destination_dir):
        for patient_id in patient_list:
            for npz_file in patient_files[patient_id]:
                shutil.move(os.path.join(npz_dir, npz_file), os.path.join(destination_dir, npz_file))
                print(f"Moved {npz_file} to {destination_dir}")
    
    # Split patients into train, test, and validation sets
    train_patients = patient_ids[:train_count]
    test_patients = patient_ids[train_count:train_count + test_count]
    val_patients = patient_ids[train_count + test_count:]
    
    # Move files to respective directories
    move_files(train_patients, train_dir)
    move_files(test_patients, test_dir)
    move_files(val_patients, val_dir)

#--------------------------
#BELOW HERE IS FOR MOVING BACK TO ONE FOLDER
# Function to move files with error handling
def move_file(src, dst):
    try:
        shutil.move(src, dst)
    except PermissionError as e:
        print(f"Permission error moving file {src} to {dst}: {e}")
    except Exception as e:
        print(f"Error moving file {src} to {dst}: {e}")

# Function to move all files back to the base directory
def move_all_to_base_dir(base_dir):
    # Directories for train, validation, and test sets
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    
    # Function to move files from a given directory to the base directory
    def move_files_from_dir(source_dir):
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(base_dir, filename)
            if os.path.isfile(source_file):
                move_file(source_file, target_file)
    
    # Move files from train, validation, and test directories
    move_files_from_dir(train_dir)
    move_files_from_dir(val_dir)
    move_files_from_dir(test_dir)
    
    # Remove empty directories after moving files
    for sub_dir in ['train', 'validation', 'test']:
        dir_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(dir_path):
            os.rmdir(dir_path)

    print("All files have been moved back to the base directory.")


if __name__ == "__main__":
    # Example usage:
    base_dir = '/proj/berzelius-2023-86/users/x_anmka/combined_myo_segs_npz'

    #If we want to specify if we want a certain patient in a certain folder. Mainly because one patient only had 7 time-points, and one 19 which we did not want in the test set.
    specified_patients = {
        "train": [],
        "validation": [],
        "test": []
    } #TODO What to do with pt39??

    train_ratio = 70
    val_ratio = 10
    test_ratio = 20



    #If we true we split the data. If false we combine the three folders back to one again.
    split_data_bool = True

    if split_data_bool:
        #move_specfic_patients(base_dir, specified_patients)
        split_data(base_dir, train_ratio=70, val_ratio=10, test_ratio=20)

    else:
        # Example usage of resetting files:
        move_all_to_base_dir(base_dir)
