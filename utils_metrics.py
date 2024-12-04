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



# Example usage
# print_npy_file('path_to_your_file.npy')
    

# Calculate MSE between output images and input images
# mse_between_folders_grouped("output_imgs", "UCF_224x224x3_PNC_FrameCorr_input_imgs")
        
# read .npy file
def print_and_save_npy_slice(file_path, output_file):
    # Load the .npy file
    data = np.load(file_path)
    
    # Print the shape of the array
    print("Shape of the array:", data.shape)
    
    # Extract the slice (55, feature number 1, 32, 32)
    slice_data = data[:, 0, :, :]
    
    # Print the slice with clean formatting
    np.set_printoptions(precision=3, suppress=True)
    print(slice_data)
    
    # Save the slice to a text file with better formatting
    with open(output_file, 'w') as f:
        for i in range(slice_data.shape[0]):
            f.write(f"Frame {i}:\n")
            f.write("[\n")
            for row in slice_data[i]:
                f.write("  " + np.array2string(row, formatter={'float_kind':lambda x: "%.3f" % x}) + "\n")
            f.write("]\n\n")

def save_diff_between_consecutive_indices(file_path, output_file):
    # Load the .npy file
    data = np.load(file_path)
    
    # Extract the slice (55, feature number 1, 32, 32)
    slice_data = data[:, 0, :, :]
    
    # Compute the differences between consecutive indices
    diffs = []
    for i in range(0, slice_data.shape[0] - 1, 2):
        diff = slice_data[i] - slice_data[i + 1]
        diffs.append(diff)
    
    # Save the differences to a text file with better formatting
    with open(output_file, 'w') as f:
        for i, diff in enumerate(diffs):
            f.write(f"Difference between frames {2*i} and {2*i+1}:\n")
            f.write("[\n")
            for row in diff:
                f.write("  " + np.array2string(row, formatter={'float_kind':lambda x: "%.3f" % x}) + "\n")
            f.write("]\n\n")


# print_and_save_npy_slice('PNC_combined_features/diving_4_combined.npy', 'output_slice.txt')
# save_diff_between_consecutive_indices('PNC_combined_features/diving_4_combined.npy', 'output_diff.txt')

