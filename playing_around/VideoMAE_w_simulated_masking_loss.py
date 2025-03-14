import os 
import cv2
import torch, torchvision, torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor, VideoMAEForPreTraining
from PIL import Image
import torch.nn as nn

def unpatchify(x, config):
    """
    Reconstruct video frames from flattened patches.
    Args:
        x (torch.Tensor): Tensor of shape [B, num_patches, patch_dim] from the decoder.
        config: Model configuration object with attributes:
            - patch_size (int): e.g., 16
            - tubelet_size (int): e.g., 2
            - image_size (int): e.g., 224
    
    Returns:
        torch.Tensor: Reconstructed video tensor of shape [B, T, 3, image_size, image_size]
    """
    patch_size = config.patch_size        # e.g., 16
    tubelet_size = config.tubelet_size      # e.g., 2
    image_size = config.image_size          # e.g., 224

    num_patches_per_frame = (image_size // patch_size) ** 2  # e.g., (224//16)^2 = 14^2 = 196
    B, N, patch_dim = x.shape
    T = N // num_patches_per_frame  # number of tubelet frames
    
    # Ensure that patch_dim equals flattened patch pixels (assumes 3 channels)
    if patch_dim != 3 * (patch_size ** 2):
        raise ValueError("Mismatch in patch dimension. Expected {} but got {}."
                         .format(3 * (patch_size ** 2), patch_dim))
    
    # Reshape into (B, T, patches_per_frame, 3, patch_size, patch_size)
    x = x.reshape(B, T, num_patches_per_frame, 3, patch_size, patch_size)
    
    # Rearrange patches into grid; assume grid is square (grid_size x grid_size patches)
    grid_size = image_size // patch_size  # e.g., 14
    # Reshape: (B, T, 3, grid_size, grid_size, patch_size, patch_size)
    x = x.reshape(B, T, 3, grid_size, grid_size, patch_size, patch_size)
    # Permute and merge patch dimensions to get (B, T, 3, image_size, image_size)
    x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, T, 3, image_size, image_size)
    return x



def read_video_from_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames: raise ValueError("Video has less frames than the required number of frames")

    # uniformly split up video into num_frames frames (e.g. if video has 160 frames and num_frames=16, then take every 10th frame, recording indices of every 10th frame)
    indices = np.linspace(0, total_frames, num_frames, endpoint=False, dtype=int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: 
            break
    
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

def main():
    video_dir = "./Kinetics-400-subset-6-videos" # NOTE: click this link to access an example: https://www.kaggle.com/datasets/ipythonx/k4testset?resource=download
    video_files = os.listdir(video_dir)
    output_dir = "./Kinetics-400-test-subset-output"
    os.makedirs(output_dir, exist_ok=True)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")  # NOTE: or can use "MCG-NJU/videomae-base"
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics") # NOTE: format is 
    model.to(device)
    model.eval()

    # Define projection layer to map from 1536 to expected patch dimension (768 for a 16x16 patch with 3 channels)
    patch_size = model.config.patch_size        # e.g., 16
    expected_patch_dim = 3 * (patch_size ** 2)     # 3*16*16 = 768
    proj = nn.Linear(1536, expected_patch_dim).to(device)

    print("FINISHED loading VideoMAE model and feature extractor. Start Processing Vids!")
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print("Processing video:", video_path)  
        frames = read_video_from_frames(video_path, num_frames=16)
        if len(frames) < 16:
            print("WARNING: Skipping video due to insufficient frames?")
            continue 
        
        videos_list = [frames] # Wrap frames in a list to create a batch of size 1.

        # Preprocess frames using the feature extractor: The extractor expects images in "THWC" format (since each PIL image is HWC),
        # but since we provide a list of PIL images, convert them to a tensor of shape (1, C=3, T, H, W) where T=16.
        inputs = feature_extractor(
            images=videos_list, 
            return_tensors="pt", 
            input_data_format="channels_last"
        )["pixel_values"].to(device)

        patch_size = model.config.patch_size
        tubelet_size = model.config.tubelet_size
        image_size = model.config.image_size  # typically 224
        num_frames = inputs.shape[1] # T
        num_patches_per_frame = (image_size // patch_size) ** 2
        seq_length = (num_frames // tubelet_size) * num_patches_per_frame

        masked = (torch.rand(inputs.shape[0], seq_length) < 0.9).bool().to(device)
        print("inputs shape:", inputs.shape)
        print("masked shape:", masked.shape)    
        with torch.no_grad():
            outputs = model(inputs, bool_masked_pos=masked)

        projected_logits = proj(outputs.logits)

        reconstructed = unpatchify(projected_logits, model.config)

        # convert reconstructed video to np array: (T, H, W, C) (move channels to last dim)
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8) # IF the model outputs in [0, 1] range, scale to [0, 255] and convert to uint8

        # save each reconstructed frame in a subfolder named after each individual video
        video_name = os.path.splitext(os.basename(video_path))[0]
        video_output_subdir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_subdir, exist_ok=True)
        for i, frame in enumerate(reconstructed):
            frame = Image.fromarray(frame)
            frame_path = os.path.join(video_output_subdir, f"frame_{i}.png")
            frame.save(frame_path)
            print(f"Saved {frame_path}")
        print("Finished processing video:", video_path)

if __name__ == "__main__":
    main()
