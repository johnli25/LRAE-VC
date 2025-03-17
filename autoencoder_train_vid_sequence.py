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
from models import PNC_Autoencoder, PNC_256Unet_Autoencoder, PNC16, TestNew, TestNew2, TestNew3, PNC_with_classification, LRAE_VC_Autoencoder
from tqdm import tqdm

# NOTE: uncomment below if you're using UCF Sports Action 
class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

test_img_names = {
    "Diving-Side_001", "Golf-Swing-Front_005", "Kicking-Front_003",
    "Lifting_002", "Riding-Horse_006", "Run-Side_001",
    "SkateBoarding-Front_003", "Swing-Bench_016", "Swing-SideAngle_006", "Walk-Front_021"
}

class VideoFrameSequenceDataset(Dataset):
    """
    A dataset that reads all frames from each video in `root_dir` and returns
    subsequences of length `seq_len`.
    Example file naming: Diving-Side_001_0.jpg, Diving-Side_001_1.jpg, ...
    """
    def __init__(self, img_dir, seq_len=20, transform=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform

        # 1) Gather all frame paths, grouped by video "prefix".
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
        # print("self.video_dict AFTER SORTING: ", self.video_dict.keys())
        
        # 3) Build a global list of (prefix, start_idx) for each valid subsequence
        self.start_samples = []  # each element is (prefix, start_frame_index_in_this_video) # NOTE: the way self.start_samples is constructed excludes the last seq_len - 1 frames of each video from being used as the starting index of a sequence

        for prefix, frame_paths in self.video_dict.items():
            num_frames = len(frame_paths)
            # We can form (num_frames - seq_len + 1) subsequences if seq_len <= num_frames
            if num_frames >= self.seq_len:
                for start_idx in range(num_frames - self.seq_len + 1):
                    self.start_samples.append((prefix, start_idx))
        
        print("samples: ", self.start_samples[:73])  # Print a few samples


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


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True,
                            choices=["PNC", "PNC_256U", "PNC16", "TestNew", 
                                     "TestNew2", "TestNew3", "PNC_NoTail", 
                                     "PNC_with_classification", "LRAE_VC"],
                            help="Model to train")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
        parser.add_argument("--epochs", type=int, default=28, help="Number of epochs to train")
        return parser.parse_args()

    args = parse_args()

    # Hyperparameters
    num_epochs = args.epochs
    batch_size = 2      # example smaller batch_size for sequences
    learning_rate = 1e-3
    seq_len = 20        # We want 20-frame subsequences
    img_height, img_width = 224, 224
    path = "TUCF_sports_action_224x224/"

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)), 
        transforms.ToTensor(),
    ])

    ### 1) Create FrameSequenceDataset
    dataset = VideoFrameSequenceDataset(
        img_dir=path,
        seq_len=seq_len,
        transform=transform
    )
    print(f"Total subsequences in dataset: {len(dataset)}, with shape = {dataset[0][0].shape}")
    test_prefixes = set(test_img_names)

    all_indices = list(range(len(dataset)))
    print("length of all_indices: ", len(all_indices))  
    train_val_indices, test_indices = [], []

    for i in tqdm(all_indices, desc="Processing indices"):
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
    train_size = int(1.0 * len(train_val_indices)) # NOTE: currently, 100% of training set kept and 0% validation set
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: pick your autoencoder model
    if args.model == "PNC16":
        model = PNC16().to(device)

    # Possibly load existing weights
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Define loss/optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    