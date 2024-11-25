import os
import numpy as np
from PIL import Image
from collections import defaultdict


def calculate_mse(image1, image2):
    """Calculate MSE between two image arrays."""
    return np.mean((image1 - image2) ** 2)

def get_prefix(filename):
    """Extract prefix from filename, assuming prefix is before the last underscore."""
    prefix = "_".join(filename.split("_")[:-1])
    return prefix

def mse_between_folders_grouped(output_folder, input_folder):
    mse_groups = defaultdict(list)
    
    # Iterate over each file in the output folder
    for filename in os.listdir(output_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Check if a corresponding file exists in the input folder
        if os.path.isfile(input_path) and os.path.isfile(output_path):
            img_input = Image.open(input_path)
            img_output = Image.open(output_path)
            
            # Ensure the images are the same size
            if img_input.size != img_output.size:
                print(f"Skipping {filename}: images have different dimensions.")
                continue
            
            # Convert images to numpy arrays
            img_input = np.array(img_input)
            img_output = np.array(img_output)
            
            # Calculate the MSE
            mse = calculate_mse(img_input, img_output)
            
            # Group by prefix
            prefix = get_prefix(filename)
            mse_groups[prefix].append(mse)
            # print("MSE for", filename, ":", mse)
        else:
            print(f"No corresponding file for {filename} in input folder.")
        
    # Print the results
    for prefix, mse in mse_groups.items():
        print(f"Average MSE for {prefix}: {np.mean(mse)}")
    

# Calculate MSE between output images and input images
mse_between_folders_grouped("output_imgs", "UCF_224x224x3_PNC_FrameCorr_input_imgs")