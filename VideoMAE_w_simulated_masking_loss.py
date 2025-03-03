import os 
import cv2
import torch, torchvision, torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor, VideoMAEForPreTraining
from PIL import Image


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
    video_dir = "./Kinetics-400-test-subset" # NOTE: click this link to access an example: https://www.kaggle.com/datasets/ipythonx/k4testset?resource=download
    video_files = os.listdir(video_dir)
    output_dir = "./Kinetics-400-test-subset-output"
    os.makedirs(output_dir, exist_ok=True)  

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("facebookresearch/video-mae-kinetics400")  # NOTE: or can use "MCG-NJU/videomae-base"
    model = VideoMAEForPreTraining.from_pretrained("facebookresearch/video-mae-kinetics400") # NOTE: format is 
    model.eval()
    device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
    print(device)
    model.to(device)

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
            input_data_format="channels_first"
        )["pixel_values"].to(device)

        patch_size = model.config.patch_size
        tubelet_size = model.config.tubelet_size
        image_size = model.config.image_size  # typically 224
        num_patches_per_frame = (image_size // patch_size) ** 2
        num_frames = inputs.shape[2] # T
        seq_length = (num_frames // tubelet_size) * num_patches_per_frame

        masked = (torch.rand(inputs.shape[0], seq_length) < 0.0).bool().to(device)
        # masked_inputs = inputs.clone()

        with torch.no_grad():
            outputs = model(inputs, bool_masked_pos=masked)

        reconstructed = model.unpatchify(outputs.decoder_pred) # reconstructed shape: (1, 3, T, H, W)

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
