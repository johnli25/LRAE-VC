import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import subprocess

def process_folder(folder_path, output_folder="ae_lstm_output_actual"):
    """
    Process all image files in folder_path. Each image is assumed to be a side-by-side concatenation
    of a ground truth (left 224x224) and a reconstructed image (right 224x224).
    The file names are expected to follow the pattern: prefix_{start_idx}_{frame_idx}.png.
    The actual frame number is computed as (start_idx + frame_idx). The combined image (gt|recon)
    is saved to output_folder using the name: prefix_{actual_frame}.png.
    
    If the file already exists, it is replaced only if the new image's MSE (between ground truth and
    reconstruction) is lower than the previously saved one.
    """
    os.makedirs(output_folder, exist_ok=True)
    best_loss_dict = {}  # key: output file path, value: best (lowest) MSE observed

    # List all PNG files in the folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
    
    for image_file in image_files:
        base_name = os.path.basename(image_file)
        name_part = base_name[:-4]  # remove the ".png" extension
        
        # Expecting a pattern like: prefix_{start_idx}_{frame_idx}
        # We use rsplit to split from the right at most 2 times.
        parts = name_part.rsplit('_', 2)
        if len(parts) < 3:
            print(f"Skipping file {image_file}: does not match the expected naming pattern.")
            continue
        
        prefix, start_idx_str, frame_idx_str = parts
        try:
            start_idx = int(start_idx_str)
            frame_idx = int(frame_idx_str)
        except ValueError as e:
            print(f"Skipping file {image_file}: error converting indices to integer: {e}")
            continue
        
        actual_frame = start_idx + frame_idx
        output_file_name = f"{prefix}_{actual_frame}.png"
        output_file_path = os.path.join(output_folder, output_file_name)
        
        # Open the image file and convert to RGB (in case it is not)
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_file}: {e}")
            continue
        
        width, height = image.size
        # Assuming the image is composed of two 224x224 halves side by side:
        if width < 448 or height < 224:
            print(f"Skipping file {image_file}: unexpected image dimensions {width}x{height}")
            continue
        
        half_width = width // 2
        # Crop left half as ground truth and right half as reconstructed output.
        gt_img = image.crop((0, 0, half_width, height))
        recon_img = image.crop((half_width, 0, width, height))
        
        # Convert both to tensors (range [0,1])
        gt_tensor = to_tensor(gt_img)
        recon_tensor = to_tensor(recon_img)
        
        # Compute the Mean Squared Error (MSE) between gt and reconstruction.
        mse_val = torch.mean((gt_tensor - recon_tensor) ** 2).item()
        
        # Check if we've already processed a frame with this actual_frame.
        # If so, only replace if the new MSE is lower.
        if output_file_path in best_loss_dict:
            if mse_val < best_loss_dict[output_file_path]:
                best_loss_dict[output_file_path] = mse_val
                combined = torch.cat((gt_tensor, recon_tensor), dim=2)  # Concatenate along width
                save_image(combined, output_file_path)
                # print(f"Replaced {output_file_path} with lower MSE: {mse_val:.6f}")
        else:
            best_loss_dict[output_file_path] = mse_val
            combined = torch.cat((gt_tensor, recon_tensor), dim=2)  # Concatenate along width
            save_image(combined, output_file_path)
            # print(f"Saved {output_file_path} with MSE: {mse_val:.6f}")


def create_videos_from_folder(folder_path, framerate=10):
    """
    Groups image files by video prefix and creates an .mp4 video for each group.
    It assumes that files in folder_path are named like: prefix_{actual_frame}.png.
    The images for each video are sorted by the actual frame number.
    """    
    video_output_folder = os.path.join(folder_path, "videos")
    os.makedirs(video_output_folder, exist_ok=True)

    # List all PNG files in the folder.
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    
    # Group files by prefix.
    groups = {}
    for f in image_files:
        try:
            prefix, frame_part = f.rsplit('_', 1)
            # Remove .png extension and convert frame number to int.
            frame_num = int(frame_part[:-4])
        except Exception as e:
            print(f"Skipping file {f}: could not parse frame number ({e}).")
            continue
        groups.setdefault(prefix, []).append((frame_num, f))
    
    # For each group, sort by frame number and use ffmpeg to create a video.
    for prefix, files in groups.items():
        files.sort(key=lambda x: x[0])
        # Create a temporary directory to copy (or symlink) files with sequential numbering.
        temp_dir = os.path.join(folder_path, "temp_video", prefix)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy (or rename) files into temp_dir with sequential names.
        for i, (_, file_name) in enumerate(files):
            src = os.path.join(folder_path, file_name)
            dst = os.path.join(temp_dir, f"{i:06d}.png")
            # You could copy or symlink; here we'll copy.
            os.system(f"cp '{src}' '{dst}'")
        
        video_path = os.path.join(video_output_folder, f"{prefix}.mp4")
        # Build ffmpeg command. Adjust the pattern if needed.
        cmd = (
            f"ffmpeg -y -framerate {framerate} -i {temp_dir}/%06d.png "
            f"-c:v libx264 -pix_fmt yuv420p '{video_path}'"
        )
        print(f"Creating video {video_path} using command:\n{cmd}")
        subprocess.call(cmd, shell=True)
        
        # Optionally, remove the temporary directory.
        os.system(f"rm -rf '{os.path.join(folder_path, 'temp_video')}'")



# Example usage:
# folder_path = **input** folder,
process_folder(folder_path="ae_PNC32_lstm_quantize8_3bits_output_test", output_folder="ae_PNC32_lstm_quantize8_3bits_frame_and_vid")
# folder_path = **output** folder,
create_videos_from_folder(folder_path="ae_PNC32_lstm_quantize8_3bits_frame_and_vid")
