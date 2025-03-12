import os
import re
import nibabel as nib

def extract_patient_number_and_time(filename):
    # Use a regular expression to extract the patient number and time point
    match = re.search(r'pt(\d+)_dt(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def collect_patient_time_points(folder_path):
    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
    
    # Dictionary to hold patient time points
    patient_time_points = {}
    
    for file in files:
        patient_number, time_point = extract_patient_number_and_time(file)
        if patient_number is not None:
            if patient_number not in patient_time_points:
                patient_time_points[patient_number] = []
            patient_time_points[patient_number].append(time_point)
    
    # Sort time points for each patient
    for patient_number in patient_time_points:
        patient_time_points[patient_number].sort()
    
    return patient_time_points

def compare_folders(folder_path_1, folder_path_2):
    
    print("Running for path 1: ", folder_path_1.split("/")[-1])
    patient_time_points_1 = collect_patient_time_points(folder_path_1)

    print("Running for path 2: ", folder_path_2.split("/")[-1])
    patient_time_points_2 = collect_patient_time_points(folder_path_2)

    # Get patient numbers for each folder
    patients_1 = set(patient_time_points_1.keys())
    patients_2 = set(patient_time_points_2.keys())

    # Patients only in folder 1
    only_in_folder_1 = patients_1 - patients_2

    # Patients only in folder 2
    only_in_folder_2 = patients_2 - patients_1

    print("Patients only in folder 1:", only_in_folder_1)
    print("Patients only in folder 2:", only_in_folder_2)

# Example usage
lee_seg = "/mnt/anmka/fenix/cdrg/TimeResolvedCT/new_LiU3D/ground_truths"
myo_seg = "/mnt/anmka/fenix/cdrg/TimeResolvedCT/myo_280524"
ct_data = "/mnt/anmka/fenix/cdrg/TimeResolvedCT/ct_as_nifti_new"


folder_path = myo_seg

print("Running for path: ", folder_path.split("/")[-1])

patient_time_points = collect_patient_time_points(folder_path)

# Sort the keys of the dictionary
sorted_keys = sorted(patient_time_points.keys())

# Create a new dictionary with sorted keys
patient_time_points = {key: patient_time_points[key] for key in sorted_keys}


# Print the lists of time points for each patient
for patient_number, time_points in patient_time_points.items():
    print(f'Patient {patient_number}: Time Points: {time_points} {len(time_points)}')

print(f"Number of patients: {len(patient_time_points)}")


#compare_folders(lee_seg, myo_seg)





###Sanity check to doulbe check volume that's on berzelius###

# nifti_file = '/mnt/anmka/fenix/cdrg/TimeResolvedCT/ct_as_nifti_new/ct_pt39_dt10.nii.gz'
# img = nib.load(nifti_file)

# # Get the data from the NIfTI file
# data = img.get_fdata()

# # Print the shape of the data
# print("Shape of the NIfTI file data:", data.shape)