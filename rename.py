import os
import random
import string

# Set the path to your folder
folder_path = './Truck'

# Get the list of files in the folder
files = os.listdir(folder_path)

# Step 1: Rename files to random names
for file in files:
    # Generate a random filename
    random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + os.path.splitext(file)[1]
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, random_name)
    
    # Rename the file to a random name
    os.rename(old_path, new_path)

# Step 2: Rename files to Image_number format
files = os.listdir(folder_path)
for index, file in enumerate(files):
    # Get the file extension
    file_extension = os.path.splitext(file)[1]
    # Create the new name as Image_number
    new_name = f'truck{index + 1}{file_extension}'
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    
    # Check if the new name already exists
    while os.path.exists(new_path):
        # If it exists, increment the number
        index += 1
        new_name = f'Image_{index + 1}{file_extension}'
        new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print("Files renamed successfully.")
