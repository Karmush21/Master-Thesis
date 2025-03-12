import subprocess

"""
Script to move a specific folder from Berzelius to the local computer.
"""

# After training is done, transfer the folder to your local computer
remote_folder = "/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_test"
local_destination = "/home/anmka"

# Construct the scp command
scp_command = ["scp", "-r", f"x_anmka@berzelius1.nsc.liu.se:{remote_folder}", local_destination]

# Execute the scp command
try:
    subprocess.run(scp_command, check=True)
    print("Folder transferred successfully.")
except subprocess.CalledProcessError as e:
    print("Error occurred while transferring folder:", e)
