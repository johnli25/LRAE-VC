import os
import math
import subprocess
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

def process_images_select_best_mse(input_folder, output_folder="processed_images_best_mse"):
    """
    Process all image files in input_folder. Each image is assumed to be a side-by-side concatenation
    of a ground truth (left 224x224) and a reconstructed image (right 224x224).
    
    Filenames are expected to follow the pattern: 
        prefix_{start_idx}_{frame_idx}_{drop}.png
    
    The actual frame number is computed as (start_idx + frame_idx). The function creates a combined image
    (concatenated ground truth and reconstruction along the width) and saves it to output_folder using the name:
        prefix_{actual_frame}.png
    
    If a file with that name already exists, it is replaced only if the new image's MSE (between ground truth 
    and reconstruction) is lower than the previously saved one.
    """
    os.makedirs(output_folder, exist_ok=True)
    best_loss_dict = {}  # key: output file path, value: best (lowest) MSE observed

    # List all PNG files in the input folder.
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".png")]
    
    for image_file in image_files:
        base_name = os.path.basename(image_file)
        name_part = base_name[:-4]  # remove the ".png" extension
        
        # Expected pattern: prefix_{start_idx}_{frame_idx}_{drop}
        parts = name_part.rsplit('_', 3)
        if len(parts) < 3:
            print(f"Skipping file {image_file}: does not match the expected naming pattern.")
            continue
        
        # Unpack parts (drop might be used for quality info)
        prefix, start_idx_str, frame_idx_str, drop = parts
        try:
            start_idx = int(start_idx_str)
            frame_idx = int(frame_idx_str)
        except ValueError as e:
            print(f"Skipping file {image_file}: error converting indices to integer: {e}")
            continue
        
        actual_frame = start_idx + frame_idx
        output_file_name = f"{prefix}_{actual_frame}.png"
        output_file_path = os.path.join(output_folder, output_file_name)
        
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_file}: {e}")
            continue
        
        width, height = image.size
        if width < 448 or height < 224:
            print(f"Skipping file {image_file}: unexpected image dimensions {width}x{height}")
            continue
        
        half_width = width // 2
        gt_img = image.crop((0, 0, half_width, height))
        recon_img = image.crop((half_width, 0, width, height))
        
        gt_tensor = to_tensor(gt_img)
        recon_tensor = to_tensor(recon_img)
        
        mse_val = torch.mean((gt_tensor - recon_tensor) ** 2).item()
        
        if output_file_path in best_loss_dict:
            if mse_val < best_loss_dict[output_file_path]:
                best_loss_dict[output_file_path] = mse_val
                combined = torch.cat((gt_tensor, recon_tensor), dim=2)
                save_image(combined, output_file_path)
                # print(f"Replaced {output_file_path} with lower MSE: {mse_val:.6f}")
        else:
            best_loss_dict[output_file_path] = mse_val
            combined = torch.cat((gt_tensor, recon_tensor), dim=2)
            save_image(combined, output_file_path)
            # print(f"Saved {output_file_path} with MSE: {mse_val:.6f}")


def create_videos_grouped_by_prefix_and_start(input_folder, output_folder, framerate=10):
    """
    Groups image files in input_folder by (prefix, start_idx) based on the naming convention:
        prefix_{start_idx}_{frame_idx}_{drop}.png
    It computes the actual frame as (start_idx + frame_idx), sorts each group by this value, and creates an .mp4 video
    for each group using ffmpeg.
    
    Videos are saved to a "videos" subfolder inside the specified output_folder.
    
    Parameters:
      input_folder (str): Folder containing the image files.
      output_folder (str): Folder where videos and temporary files will be stored.
      framerate (int): Frame rate for the output video.
    """
    # Create output folder and subfolder for videos.
    video_output_folder = os.path.join(output_folder, "videos")
    os.makedirs(video_output_folder, exist_ok=True)
    
    # List all PNG files in the input folder.
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    
    # Group files by (prefix, start_idx)
    groups = {}
    for f in image_files:
        try:
            # Expected pattern: prefix_{start_idx}_{frame_idx}_{drop}.png
            prefix, start_idx_str, frame_idx_str, drop = f[:-4].rsplit('_', 3)
            start_idx = int(start_idx_str)
            frame_idx = int(frame_idx_str)
            actual_frame = start_idx + frame_idx
        except Exception as e:
            print(f"Skipping file {f}: could not parse filename ({e}).")
            continue
        key = (prefix, start_idx)
        groups.setdefault(key, []).append((actual_frame, f))
    
    # Process each group: sort by actual frame and create a video.
    for (prefix, start_idx), frames in groups.items():
        frames.sort(key=lambda x: x[0])
        # Create a temporary directory inside output_folder.
        temp_dir = os.path.join(output_folder, "temp_video", f"{prefix}_{start_idx}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy files to temp_dir with sequential numbering (required by ffmpeg).
        for i, (_, file_name) in enumerate(frames):
            src = os.path.join(input_folder, file_name)
            dst = os.path.join(temp_dir, f"{i:06d}.png")
            os.system(f"cp '{src}' '{dst}'")
        
        video_file = os.path.join(video_output_folder, f"{prefix}_{start_idx}.mp4")
        cmd = (
            f"ffmpeg -y -framerate {framerate} -i {temp_dir}/%06d.png "
            f"-c:v libx264 -pix_fmt yuv420p '{video_file}'"
        )
        print(f"Creating video: {video_file}\nCommand: {cmd}")
        subprocess.call(cmd, shell=True)
        
        # Remove the temporary directory.
        os.system(f"rm -rf '{os.path.join(output_folder, 'temp_video')}'")


# Example usage:
# First, process the images to select the best (lowest MSE) version:
# process_images_select_best_mse(input_folder="ae_lstm_output_test_realistic", output_folder="processed_images_best_mse")

# Then, create videos from the processed images grouping by (prefix, start_idx):
create_videos_grouped_by_prefix_and_start(input_folder="ae_lstm_output_test_realistic", output_folder="PNC32_AE_lstm_output_test_realistic_frame_and_vids", framerate=10)
