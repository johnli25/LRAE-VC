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

torch.cuda.empty_cache()

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

    def __len__(self):
        return len(self.start_samples)

    def __getitem__(self, idx):
        """
        Returns a Tensor of shape (seq_len, 3, H, W).
        """
        prefix, start_idx = self.start_samples[idx]
        frame_paths = self.video_dict[prefix]
        
        frames = []
        for i in range(start_idx, start_idx + self.seq_len):
            fpath = frame_paths[i]
            img = Image.open(fpath).convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        # Stack into a single tensor of shape (seq_len, 3, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        
        return frames_tensor, prefix, start_idx  # (frames_tensor, video_prefix, start_frame_index)


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
    total_mse = 0.0
    model = model.module if isinstance(model, nn.DataParallel) else model

    model.eval()
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


    # Save the reconstructed images
    output_path = "ae_analysis_reconstructed_imgs"
    os.makedirs(output_path, exist_ok=True)
    for i, (images, filenames) in enumerate(test_loader):
        images = images.to(device)
        encoding = model.encode(images)
        decoded_reconstruction = model.decode(encoding)

        # Save the original and reconstructed images
        for j in range(len(images)):
            output_np = decoded_reconstruction[j].detach().permute(1, 2, 0).cpu().numpy()            
            plt.imsave(os.path.join(output_path, filenames[j]), output_np)

    return avg_mse

def main_ae_lstm(model_path, dataloader, criterion, drop=0, quantize=False, bidirectional=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM_AE(total_channels=32, hidden_channels=32, ae_model_name="PNC32", bidirectional=bidirectional).to(device)
    
    # NOTE: Load the state_dict and remove "module." prefix if necessary
    state_dict = torch.load(model_path)
    if any(key.startswith("module.") for key in state_dict.keys()):
        # Remove "module." prefix
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    # Remove deprecated keys
    deprecated_keys = ["conv_lstm.forward_cell.drift", "conv_lstm.backward_cell.drift"]
    for key in deprecated_keys:
        if key in state_dict:
            del state_dict[key]

    # Load the filtered state_dict
    model.load_state_dict(state_dict)

    total_mse, total_frames = 0.0, 0
    model = model.module if isinstance(model, nn.DataParallel) else model
    # NOTE: end filtering state_dict/model 

    model.eval()
    with torch.no_grad():
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            frames_tensor = frames.to(device)
            outputs, _, _ = model(x_seq=frames_tensor, drop=drop, quantize=quantize)
            
            for b in range(frames_tensor.size(0)): # for each batch
                for t in range(frames_tensor.size(1)): # for each frame in video sequence
                    gt = frames_tensor[b, t]
                    pred = outputs[b, t]
                    frame_mse = nn.functional.mse_loss(pred, gt).item()
                    total_mse += frame_mse
                    total_frames += 1

    avg_mse = total_mse / total_frames
    print(f"Average MSE: {avg_mse:.4f}")


    # Save the reconstructed images
    # output_path = "ae_analysis_reconstructed_imgs"
    # os.makedirs(output_path, exist_ok=True)
    # for i, (frames, prefix_, start_idx_) in enumerate(dataloader):
    #     frames_tensor = frames.to(device)
    #     outputs, _, _ = model(x_seq=frames_tensor, drop=drop, quantize=quantize)

    #     for b in range(frames_tensor.size(0)): # for each batch
    #         for t in range(frames_tensor.size(1)): # for each frame in video sequence
    #             gt = frames_tensor[b, t]
    #             pred = outputs[b, t]
    #             output_np = pred.detach().permute(1, 2, 0).cpu().numpy()            
    #             plt.imsave(os.path.join(output_path, f"{prefix_[b]}_{start_idx_}_{t}.png"), output_np)
    
    return avg_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PNC")
    parser.add_argument("--model", type=str, default="PNC16", help="Model name")
    parser.add_argument("--model_path", type=str, default="models/PNC16.pth", help="Path to the model")
    parser.add_argument("--bidirectional", action="store_true", help="Enable Bidirectional (ONLY FOR ConvLSTM)!!")

    args = parser.parse_args()

    img_height, img_width = 224, 224
    path = "TUCF_sports_action_224x224/"
    batch_size = 8
    seq_len = 20
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
        dataset = VideoFrameSequenceDataset(
            img_dir=path,
            seq_len=seq_len,
            transform=transform,
            step_thru_frames=1
        )

        test_prefixes = set(test_img_names)

        all_indices = list(range(len(dataset)))
        print("length of all_indices: ", len(all_indices))  
        train_val_indices, test_indices = [], []

        for i in all_indices:
            # Just use dataset.start_samples (it's already a list of (prefix, start_idx))
            sample_prefix, _ = dataset.start_samples[i]
            
            if sample_prefix in test_prefixes:
                test_indices.append(i)
            else:
                train_val_indices.append(i)

        print(f"Test subsequences: {len(test_indices)}")
        # Shuffle train_val_indices
        np.random.shuffle(train_val_indices)

        # Now pick 90% for train, 10% for val
        train_size = int(0.9 * len(train_val_indices))
        val_size = len(train_val_indices) - train_size

        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        print(f"Train subsequences: {len(train_indices)}")
        print(f"Val   subsequences: {len(val_indices)}")

        # Create Subsets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print("Size of Dataloaders  ---  train_loader:", len(train_loader), "val_loader:", len(val_loader), "test_loader:", len(test_loader))       
        main_ae_lstm(args.model_path, test_loader, criterion, drop=0.0, quantize=False, bidirectional=args.bidirectional)
    else:
        raise ValueError("Invalid model name. Choose either 'PNC32' or 'conv_lstm_PNC32_ae'.")
    