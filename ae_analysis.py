import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import PNC16, PNC32, ConvLSTM_AE
from tqdm import tqdm
import random
import torchvision.utils as vutils
import zlib 
import shutil
from pytorch_msssim import ssim
import time, csv

class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

test_img_names = {
    "Diving-Side_001", 
    # "Golf-Swing-Front_005", 
    # "Kicking-Front_003", # just use these video(s) for temporary results
    # "Lifting_002", "Riding-Horse_006", "Run-Side_001",
    # "SkateBoarding-Front_003", "Swing-Bench_016", "Swing-SideAngle_006", "Walk-Front_021"
}

class VideoFrameSequenceDataset(Dataset):
    """
    A dataset that reads all frames from each video in `root_dir` and returns
    subsequences of length `seq_len`.
    Example file naming: Diving-Side_001_0.jpg, Diving-Side_001_1.jpg, ...
    """
    def __init__(self, img_dir, seq_len=20, step_thru_frames=1, transform=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.step_thru_frames = step_thru_frames    
        self.transform = transform

        # 1) gather all frame paths, grouped by video "prefix".
        #    We'll define the "prefix" as everything except the frame index at the end.
        #    E.g. "Diving-Side_001" is the prefix; the frame index is after the last underscore.
        self.video_dict = {}

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
                continue

            fpath = os.path.join(img_dir, fname)
            
            # We assume filenames look like "Diving-Side_001_0.jpg" and split out the prefix from the final frame index
            # the approach is to split on the last underscore: e.g. "Diving-Side_001_0.jpg" => prefix = "Diving-Side_001", frame_idx = "0"
            parts = fname.rsplit('_', maxsplit=1) # But do this to be safe in case there's underscores in the prefix itself
            if len(parts) < 2:
                continue # If for some reason there's no underscore, skip

            prefix = parts[0] # e.g. "Diving-Side_001"
            if prefix not in self.video_dict:
                self.video_dict[prefix] = []
            self.video_dict[prefix].append(fpath)
        
        # 2) For each prefix (video), sort by frame index because we want them in chronological order
        for prefix in self.video_dict:
            # We can parse the frame index from the filenames or use a sorted approach.
            # If your naming is strictly: prefix_index.jpg, a simple numeric sort can work:
            def extract_frame_idx(fpath):
                # e.g. fpath = ".../Diving-Side_001_13.jpg"
                fname_only = os.path.basename(fpath)
                idx_str = fname_only.rsplit('_', 1)[-1].split('.')[0]
                return int(idx_str) if idx_str.isdigit() else -1

            self.video_dict[prefix].sort(key=extract_frame_idx)
        
        # 3) Build a global list of (prefix, start_idx) for each valid subsequence
        self.start_samples = []  # each element is (prefix, start_frame_index_in_this_video) # NOTE: the way self.start_samples is constructed excludes the last seq_len - 1 frames of each video from being used as the starting index of a sequence

        for prefix, frame_paths in self.video_dict.items():
            num_frames = len(frame_paths)
            # We can form (num_frames - seq_len + 1) subsequences if seq_len <= num_frames
            if num_frames >= self.seq_len:
                # Generate start indices with step_between_clips
                start_indices = list(range(0, num_frames - self.seq_len + 1, self.step_thru_frames))

                # Ensure the last valid subsequence is included
                if not start_indices: print(f"start_indices is SOMEHOW empty?! Video/prefix: {prefix}, num_frames: {num_frames}") # just a sanity check
                if start_indices[-1] != num_frames - self.seq_len:
                    start_indices.append(num_frames - self.seq_len)

                # Add all start indices to self.start_samples
                self.start_samples.extend((prefix, idx) for idx in start_indices)



class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Use the same image for both input and ground truthx
        return image, self.img_names[idx]  # (image, same_image_as_ground_truth, img filename)
    

def main_ae_pnc(model_path, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PNC32().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    total_mse = 0.0
    model = model.module if isinstance(model, nn.DataParallel) else model

    # NOTE: this is basically eval_autoencoder() in autoencoder_train.py
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            encoding = model.encode(images)
            decoding = model.decode(encoding)
            mse = criterion(decoding, images)
            total_mse += mse.item()
    avg_mse = total_mse / len(test_loader)
    print(f"Average MSE: {avg_mse:.4f}")
    return avg_mse

def main_ae_lstm(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM_AE().to(device)
    model.load_state_dict(torch.load(model_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PNC")
    parser.add_argument("--model", type=str, default="PNC16", help="Model name")
    parser.add_argument("--model_path", type=str, default="models/PNC16.pth", help="Path to the model")
    args = parser.parse_args()

    img_height, img_width = 224, 224
    path = "TUCF_sports_action_224x224/"
    batch_size = 16
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)), # NOTE: not needed if dataset already resized (e.g. to 224x224)
        transforms.ToTensor(),
    ])
    criterion = nn.MSELoss()

    if args.model == "PNC32":
        model = PNC32()
        dataset = ImageDataset(path, transform=transform)

        test_indices = [
            i for i in range(len(dataset))
            if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
        ]
        train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

        # Split train_val_indices into train and validation
        np.random.shuffle(train_val_indices)
        train_size = int(0.9 * len(train_val_indices))
        val_size = len(train_val_indices) - train_size

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # apply data augmentation to the train data
        train_dataset.dataset.transform = transform
        val_dataset.dataset.transform = transform        
        test_dataset.dataset.transform = transform

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(type(test_loader))
        main_ae_pnc(args.model_path, test_loader, criterion)

    elif args.model == "conv_lstm_PNC32_ae":
        model = ConvLSTM_AE(total_channels=32, hidden_channels=32, ae_model_name="PNC32", bidirectional=args.bidirectional)

        