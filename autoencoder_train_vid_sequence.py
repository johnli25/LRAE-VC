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
from models import PNC16, PNC16_LSTM_AE, ConvLSTM_AE, PNC_Autoencoder, PNC_256Unet_Autoencoder, TestNew, TestNew2, TestNew3, PNC_with_classification, LRAE_VC_Autoencoder
from tqdm import tqdm
import random
import torchvision.utils as vutils


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
    def __init__(self, img_dir, seq_len=20, step_thru_frames=1, transform=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.step_thru_frames = step_thru_frames    
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
                # Generate start indices with step_between_clips
                start_indices = list(range(0, num_frames - self.seq_len + 1, self.step_thru_frames))

                # Ensure the last valid subsequence is included
                if not start_indices: print(f"start_indices is SOMEHOW empty?! Video/prefix: {prefix}, num_frames: {num_frames}") # just a sanity check
                if start_indices[-1] != num_frames - self.seq_len:
                    start_indices.append(num_frames - self.seq_len)

                # Add all start indices to self.start_samples
                self.start_samples.extend((prefix, idx) for idx in start_indices)
        
        # print("samples: ", self.start_samples[:73])  # Print a few samples

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


def plot_train_val_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')
    
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig('train_val_loss_curve.png', dpi=300)
    plt.show()


def train(ae_model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name=None, max_drops=0):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    os.makedirs("ae_lstm_output_train", exist_ok=True)
    best_val_losses = {}
    if max_drops > 0: 
        drops = 0  # should be 1 less than the drops you ACTUALLY want to start at
    
    for epoch in range(num_epochs):
        # steadily increase the max # of drops every 2 epochs
        if max_drops > 0 and epoch % 2 == 0:
            drops = min(drops + 1, max_drops) 
            if hasattr(ae_model, "module"): ae_model.module.drop = drops
            else: ae_model.drop = drops
            print(f"Tail Dropouts = {drops} for epoch {epoch}")

        ae_model.train()
        epoch_loss = 0.0    
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch")):
            frames_tensor = frames.to(device)
            
            optimizer.zero_grad()
            recon = ae_model(frames_tensor)  # -> (batch_size, seq_len, 3, 224, 224)
            loss = criterion(recon, frames_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # NOTE: save reconstructed img into folder for sanity check purposes
            # if batch_idx % 10 == 0:  # Save every 10 batches
            #     for seq_idx in range(frames_tensor.size(0)):  # Iterate over each batch in batch_size 
            #         frame_idx = random.randint(0, frames_tensor.size(1) - 1)  # Pick a random frame in the sequence
            #         frame_input = frames_tensor[seq_idx, frame_idx].cpu()
            #         frame_output = recon[seq_idx, frame_idx].cpu()

            #         # Save original and reconstructed frame side by side
            #         combined = torch.cat([frame_input, frame_output], dim=2)  # Concatenate along width
            #         vutils.save_image(combined, f"ae_lstm_output_train/{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}.png")

            #         print(f"Saved train sample: ae_lstm_output_train/{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}.png")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        val_loss = evaluate(ae_model, val_loader, criterion, device, save_sample="val")
        val_losses.append(val_loss)

        # Check and update best validation loss for the current dropout level
        if max_drops > 0:
            # Use current_drop as key; if not present, use infinity
            if drops not in best_val_losses or val_loss < best_val_losses[drops]:
                best_val_losses[drops] = val_loss
                if hasattr(ae_model, "module"):
                    torch.save(ae_model.module.state_dict(), f"{model_name}_drop_{drops}_features_best_val_weights.pth")
                else:
                    torch.save(ae_model.state_dict(), f"{model_name}_drop_{drops}_features_best_val_weights.pth")
                print(f"New best model for dropout {drops} saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        else: # for NO dropout 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ae_model.state_dict(), f"{model_name}_best_val_weights.pth")
                print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            

    plot_train_val_loss(train_losses, val_losses)

    # Final test/evaluation
    test_loss = evaluate(ae_model, test_loader, criterion, device, save_sample="test")
    print(f"Test Loss: {test_loss:.4f}")



def evaluate(ae_model, dataloader, criterion, device, save_sample=False):
    ae_model.eval()
    running_loss = 0.0
    os.makedirs("ae_lstm_output_test", exist_ok=True)
    os.makedirs("ae_lstm_output_val", exist_ok=True)


    # Test/Evaluate with dropout of 10/16 features 
    if hasattr(ae_model, "module"):
        ae_model.module.drops = 10
    else:
        ae_model.drops = 10
    
    with torch.no_grad():
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            frames_tensor = frames.to(device)
            outputs = ae_model(frames_tensor)
            loss = criterion(outputs, frames_tensor)
            running_loss += loss.item()
            
            # Save a random frame from each sequence in the batch
            for seq_idx in range(frames_tensor.size(0)):  # Iterate over each seq in the batch
                frame_idx = random.randint(0, frames_tensor.size(1) - 1)  # Pick a random frame in the sequence
                frame_input = frames_tensor[seq_idx, frame_idx].cpu()
                frame_output = outputs[seq_idx, frame_idx].cpu()
                
                # Save original and reconstructed frame side by side
                combined = torch.cat([frame_input, frame_output], dim=2)  # Concatenate along width
                if save_sample == "test":
                    vutils.save_image(combined, f"ae_lstm_output_test/{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}.png")
                    # print(f"Saved test sample: ae_lstm_output_test/{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}.png")
                else:
                    vutils.save_image(combined, f"ae_lstm_output_val/{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}.png")
                    # print(f"Saved val sample: ae_lstm_output_val/{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}.png")

    return running_loss / len(dataloader)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True,
                            choices=["PNC", "PNC16", "PNC16_lstm_ae", "conv_lstm_ae",
                                    "LRAE-VC", "TestNew", "TestNew2", "TestNew3"],
                            help="Model to train")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
        parser.add_argument("--epochs", type=int, default=28, help="Number of epochs to train")
        parser.add_argument("--drops", type=int, default=0, help="MAX dropout to enforce")
        return parser.parse_args()

    args = parse_args()

    # Hyperparameters
    num_epochs = args.epochs
    drops = args.drops
    batch_size = 16     
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
        transform=transform,
        step_thru_frames=2
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
    # Below is a print sanity check: 
    # for i in range(5): print("First 5 dataset elems:", dataset[train_val_indices[i]][1], dataset[train_val_indices[i]][2])

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: pick your autoencoder model
    if args.model == "PNC16":
        model = PNC16()
    elif args.model == "PNC16_lstm_ae":
        model = PNC16_LSTM_AE(hidden_dim=128, num_layers=2)
    elif args.model == "conv_lstm_ae": # currently based on PNC16, which is a 16-feature/channel (for encode) model
        model = ConvLSTM_AE(total_channels=16, hidden_channels=32, drop=0, use_predictor=False)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)    


    # Possibly load existing weights
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Define loss/optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name=args.model, max_drops=drops)
    # save
    if drops > 0:
        torch.save(model.state_dict(), f"{args.model}_dropUpTo_{drops}_features_final_weights.pth")
    else: # no dropout OR original model
        torch.save(model.state_dict(), f"{args.model}_final_weights.pth")
