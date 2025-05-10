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

# NOTE: uncomment below if you're using UCF Sports Action 
class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

test_img_names = {
    # "Diving-Side_001", 
    "Golf-Swing-Front_005", 
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

    
def psnr(mse):
    return -10 * np.log10(mse) # NOTE: this is the simplified PSNR formula
    return 20 * np.log10(1) - 10 * np.log10(mse)


def train(ae_model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name=None, max_drops=0, quantize=False):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_val_losses = {}
    if max_drops > 0: 
        drops = -1  # should be 1 less than the drops you ACTUALLY want to start at
    
    for epoch in range(num_epochs):
        # steadily increase the max # of drops every 2 epochs
        if max_drops > 0 and epoch % 2 == 0:
            drops = min(drops + 1, max_drops) 
            if hasattr(ae_model, "module"): ae_model.module.drop = drops
            else: ae_model.drop = drops
            print(f"Tail Dropouts = {drops} for epoch {epoch}")

        ae_model.train()
        epoch_loss = 0.0    
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", unit="batch")):
            frames_tensor = frames.to(device)
            optimizer.zero_grad()

            # drop_val = random.randint(0, drops) if max_drops > 0 else 0 # TODO: INVESTIGATE this line? Is this necessary at all? OR should I directly pass in 'drops'?
            drop_val = drops if max_drops > 0 else 0
            recon, _, _ = ae_model(x_seq=frames_tensor, drop=drop_val, quantize=quantize)  # -> (batch_size, seq_len, 3, 224, 224)

            total_loss = criterion(recon, frames_tensor)
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        val_loss, _, _ = evaluate(ae_model, val_loader, criterion, device, save_sample=None, quantize=quantize) # set to "val" if needed
        val_losses.append(val_loss)

        # Check and update best validation loss for the current dropout level
        if max_drops > 0:
            # Use current_drop as key; if not present, use infinity
            if drops not in best_val_losses or val_loss < best_val_losses[drops]:
                best_val_losses[drops] = val_loss
                if hasattr(ae_model, "module"):
                    torch.save(ae_model.module.state_dict(), f"{model_name}_drop_{drops}_features_best_val_weights.pth")
                    print(f"Saved model weights for {model_name}_drop_{drops}__features_best_val_weights.pth")
                else:
                    torch.save(ae_model.state_dict(), f"{model_name}_drop_{drops}_features_best_val_weights.pth")
                    print(f"Saved model weights for {model_name}_drop_{drops}_features_best_val_weights.pth")
                print(f"New best model for dropped features = {drops} saved at epoch {epoch} with validation loss: {val_loss:.4f}")
        else: # for NO dropout 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ae_model.state_dict(), f"{model_name}_best_val_weights.pth")
                print(f"New best model saved at epoch {epoch} with validation loss: {val_loss:.4f} as {model_name}_best_val_weights.pth")
            
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
    # plot_train_val_loss(train_losses, val_losses)
        
    # Final test/evaluation
    test_loss, test_psnr, test_ssim = evaluate(ae_model, test_loader, criterion, device, save_sample="test", drop=0, quantize=quantize) # do constant number of drops 
    print(f"Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}")


def evaluate(ae_model, dataloader, criterion, device, save_sample=None, drop=0, quantize=False):
    
    ae_model.eval()
    total_mse, total_ssim, total_frames = 0.0, 0.0, 0

    output_dir = "ae_lstm_output_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            frames_tensor = frames.to(device)
            outputs, _, _ = ae_model(x_seq=frames_tensor, drop=drop, quantize=quantize)
            
            for b in range(frames_tensor.size(0)): # for each batch
                for t in range(frames_tensor.size(1)): # for each frame in video sequence
                    gt = frames_tensor[b, t]
                    pred = outputs[b, t]
                    frame_mse = nn.functional.mse_loss(pred, gt).item()
                    total_mse += frame_mse

                    # SSIM computation NOTE: add batch + channel dimension)
                    frame_ssim = ssim(
                        pred.unsqueeze(0), # shape: (1, 3, H, W)
                        gt.unsqueeze(0), # shape: (1, 3, H, W)
                        data_range=1.0,
                        size_average=False
                    )
                    total_ssim += frame_ssim.sum().item()

                    total_frames += 1

                    ##### NOTE: "intermission" function: print approx byte size of compressed latent features. THIS DOES NOT ACTUALLY AFFECT TRAINING/EVAL NOR COMPRESS THE LATENT FEATURES via quantization. 
                    # frame_latent = ae_model.module.encoder(gt.unsqueeze(0)) if hasattr(ae_model, "module") else ae_model.encode(gt.unsqueeze(0))
                    # if quantize > 0:
                    #     features_cpu = frame_latent.squeeze(0).detach().cpu().numpy()
                    #     features_uint8 = (features_cpu * 7).astype(np.uint8)  # Convert to uint8
                    #     compressed = zlib.compress(features_uint8.tobytes())
                    #     latent_num_bytes = len(compressed)
                    #     print(f"[Simulated Compression] Frame {b}_{t} compressed size (quantized to uint8): {latent_num_bytes} bytes "
                    #         f"(Original shape: {tuple(frame_latent.shape)})")
                    # else:
                    #     features_cpu = frame_latent.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    #     compressed = zlib.compress(features_cpu.tobytes())
                    #     latent_num_bytes = len(compressed)
                    #     print(f"[Simulated Compression] Frame {b}_{t} compressed size (float32): {latent_num_bytes} bytes "
                    #         f"(Original shape: {tuple(frame_latent.shape)})")
                    ##### end intermission function

                    if save_sample:
                        save_dir = output_dir
                        file_name = f"{prefix_[b]}_{start_idx_[b]}_{t}.png"
                        file_path = os.path.join(save_dir, file_name)
                        combined_frame = torch.cat((gt.cpu(), pred.cpu()), dim=2)
                        vutils.save_image(combined_frame, file_path)

    avg_mse = total_mse / total_frames
    avg_psnr = 10 * np.log10(1.0 / avg_mse)
    avg_ssim = total_ssim / total_frames

    return avg_mse, avg_psnr, avg_ssim


def evaluate_consecutive(ae_model, dataloader, criterion, device, drop=0, quantize=False, consecutive=0):
    ae_model.eval()
    total_mse, total_ssim, total_dropped_frames = 0.0, 0.0, 0

    output_dir = "ae_lstm_output_consecutive" 
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    saved_frames = set()  # keys will be (prefix, true_frame)

    with torch.no_grad():
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            frames_tensor = frames.to(device)
            outputs, _, drop_levels = ae_model(
                x_seq=frames_tensor, drop=drop,
                eval_consecutive=consecutive, quantize=quantize
            )

            for b in range(frames_tensor.size(0)):
                for t in range(frames_tensor.size(1)):
                    if len(drop_levels) > 0 and drop_levels[b][t] != 0:
                        gt = frames_tensor[b, t]
                        pred = outputs[b, t]
                        # frame_mse = nn.functional.mse_loss(pred, gt).item()
                        frame_mse = criterion(pred, gt).mean().item()  # Compute MSE for the frame
                        total_mse += frame_mse

                        # SSIM computation NOTE: add batch + channel dimension)
                        frame_ssim = ssim(
                            pred.unsqueeze(0), # shape: (1, 3, H, W)
                            gt.unsqueeze(0), # shape: (1, 3, H, W)
                            data_range=1.0,
                            size_average=False
                        )
                        total_ssim += frame_ssim.sum().item()

                        total_dropped_frames += 1

            # NOTE: Following code block purpose is ONLY TO SAVE true image and reconstructed image side by side.
            for b in range(frames_tensor.size(0)):
                true_start = start_idx_[b]
                for t in range(frames_tensor.size(1)):
                    true_frame = true_start + t
                    key = (str(prefix_[b]).strip(), int(start_idx_[b]) + t)
                    drop_val = drop_levels[b][t]

                    # NOTE: Only save if drop > 0 -OR- if it's the very first frame
                    if drop_val == 0 and not (b == 0 and t == 0):
                        continue
                    if key in saved_frames: # NOTE: skip if we've already saved this frame!
                        continue
                    saved_frames.add(key)

                    file_name = f"{prefix_[b]}_{true_frame}_drop{drop_val}.png"
                    file_path = os.path.join(output_dir, file_name)
                    frame_input = frames_tensor[b, t].cpu()
                    frame_output = outputs[b, t].cpu()
                    combined_frame = torch.cat((frame_input, frame_output), dim=2)
                    vutils.save_image(combined_frame, file_path)

    avg_mse = total_mse / total_dropped_frames
    avg_psnr = 10 * np.log10(1.0 / avg_mse)
    avg_ssim = total_ssim / total_dropped_frames

    return avg_mse, avg_psnr, avg_ssim


def evaluate_realistic(ae_model, dataloader, criterion, device, input_drop=32, quantize=False):
    ae_model.eval()
    running_loss = 0.0
    output_dir = "ae_lstm_output_test_realistic"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs("ae_lstm_output_test_realistic", exist_ok=True)

    with torch.no_grad():
        for batch_idx, (frames, prefix_, start_idx_) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            frames_tensor = frames.to(device)
            outputs, _, drop_levels = ae_model(frames_tensor, input_drop, eval_real=True, quantize=quantize) # NOTE: returns reconstructed frames of shape (batch_size, seq_len, 3, 224, 224) AND imputed latents of shape (batch_size, seq_len, 16, 32, 32)
            loss = criterion(outputs, frames_tensor)
            running_loss += loss.item()

            # Iterate over every sequence in the batch and every frame in the sequence.
            for seq_idx in range(frames_tensor.size(0)):
                for frame_idx in range(frames_tensor.size(1)):
                    # use the drop level for this sample and timestep 
                    curr_drop = drop_levels[seq_idx][frame_idx]
                    file_name = f"{prefix_[seq_idx]}_{start_idx_[seq_idx]}_{frame_idx}_drop={curr_drop}.png"
                    file_path = os.path.join("ae_lstm_output_test_realistic", file_name)

                    frame_input = frames_tensor[seq_idx, frame_idx].cpu()
                    frame_output = outputs[seq_idx, frame_idx].cpu()
                    
                    # Concatenate original and reconstructed frame side by side.
                    combined_frame = torch.cat((frame_input, frame_output), dim=2)  # Concatenate along the width dimension
                    vutils.save_image(combined_frame, file_path)

    return running_loss / len(dataloader)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True,
                            choices=["conv_lstm_PNC16_ae", "conv_lstm_PNC32_ae"],
                            help="Model to train")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
        parser.add_argument("--epochs", type=int, default=28, help="Number of epochs to train")
        parser.add_argument("--drops", type=int, default=0, help="MAX dropout to enforce")
        parser.add_argument("--quantize", type=int, default=0, help="Quantize latent features by how many bits/levels")
        parser.add_argument("--bidirectional", action="store_true", help="Enable Bidirectional ConvLSTM")
        return parser.parse_args()

    args = parse_args()

    # Hyperparameters
    num_epochs = args.epochs
    drops = args.drops
    batch_size = 32     
    learning_rate = 1e-3
    seq_len = 20
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def convertFromDataParallelNormal(checkpoint):
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key[len("module."):] if key.startswith("module.") else key
            new_checkpoint[new_key] = value
        return new_checkpoint

    def convertFromNormalToDataParallel(checkpoint):
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = "module." + key if not key.startswith("module.") else key
            new_checkpoint[new_key] = value
        return new_checkpoint

    def updateCheckpointWithNewKeys(model, checkpoint):
        model_dict = model.state_dict()
        missing_keys = set(model_dict.keys()) - set(checkpoint.keys())
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        for key in missing_keys:
            print("Add missing key with default initialization:", key)
            checkpoint[key] = model_dict[key]
        return checkpoint

    # --- Load checkpoint with automatic conversion ---
    def loadPretrainedModel(model, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Auto-detect if we should use DataParallel based on GPU count.
        use_dataparallel = (torch.cuda.device_count() >= 2)
        
        # Determine the checkpoint's key format.
        checkpoint_keys = list(checkpoint.keys())
        if not checkpoint_keys:
            raise ValueError("Loaded checkpoint has no keys.")
        checkpoint_is_dp = checkpoint_keys[0].startswith("module.")

        # Convert keys only if necessary.
        if use_dataparallel and not checkpoint_is_dp:
            print("Converting checkpoint from normal to DataParallel format.")
            checkpoint = convertFromNormalToDataParallel(checkpoint)
        elif (not use_dataparallel) and checkpoint_is_dp:
            print("Converting checkpoint from DataParallel to normal format.")
            checkpoint = convertFromDataParallelNormal(checkpoint)
        else:
            print("Checkpoint format matches the current model setup.")

        checkpoint = updateCheckpointWithNewKeys(model, checkpoint)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded model weights from {model_path}")
        return model

    # Example: pick your autoencoder model
    if args.model == "PNC16":
        model = PNC16()
    elif args.model == "conv_lstm_PNC16_ae":
        model = ConvLSTM_AE(total_channels=16, hidden_channels=32, ae_model_name="PNC16")
    elif args.model == "conv_lstm_PNC32_ae":
        model = ConvLSTM_AE(total_channels=32, hidden_channels=32, ae_model_name="PNC32", bidirectional=args.bidirectional)
        print("model: ConvLSTM_AE with PNC32", model)

    # --- Model Initialization ---
    model = model.to(device)

    # Wrap with DataParallel if two or more GPUs are available.
    if torch.cuda.device_count() >= 2:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # Load pretrained checkpoint, converting keys as needed.
    if args.model_path:
        model = loadPretrainedModel(model, args.model_path, device)

    # Define loss/optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # NOTE: uncomment below to train (and save) the model
    # train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name=args.model, max_drops=drops, quantize=args.quantize)
    # if drops > 0:
    #     torch.save(model.state_dict(), f"{args.model}_dropUpTo_{drops}_features_final_weights.pth")
    #     print(f"Model saved as {args.model}_dropUpTo_{drops}_features_final_weights.pth")
    # else: # no dropout OR original model
    #     torch.save(model.state_dict(), f"{args.model}_final_weights.pth")
    #     print(f"Model saved as {args.model}_final_weights.pth")

    # NOTE: uncomment below to evaluate the model under {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}% drops in one go! 
    tail_len_drops = [30] # [0, 3, 6, 10, 13, 16, 19, 22, 26, 28]  # NOTE: For consecutive tail_len drops, DO NOT add 0 drops here!!!
    results = []  # List to store results for CSV

    for consecutive in [1, 3, 5]:  # Only evaluate for consecutive=1, 3, and 5
        print("consecutive: ", consecutive)
        for drop in tail_len_drops:
            if drop == 0:
                final_test_loss, final_test_psnr, final_ssim = evaluate(
                    model, test_loader, criterion, device, save_sample=None, drop=drop, quantize=args.quantize
                )
            else:
                final_test_loss, final_test_psnr, final_ssim = evaluate_consecutive(
                    model, test_loader, criterion, device, drop=drop, quantize=args.quantize, consecutive=consecutive
                )  # NOTE: IMPORTANT: Do NOT add 0 drops here!!!
            
            # Append results to the list
            results.append({
                "consecutive": consecutive,
                "tail_len_drop": drop,
                "mse": final_test_loss,
                "psnr": final_test_psnr,
                "ssim": final_ssim
            })
            
            print(f"Final Test Loss For evaluation: {final_test_loss:.6f} and PSNR: {final_test_psnr:.6f} and SSIM:{final_ssim} for tail_len_drops = {drop}")

    # # Save results to a CSV file
    # csv_file = "CASTR_results.csv"
    # with open(csv_file, mode="w", newline="") as file:
    #     writer = csv.DictWriter(file, fieldnames=["consecutive", "tail_len_drop", "mse", "psnr", "ssim"])
    #     writer.writeheader()  # Write the header row
    #     writer.writerows(results)  # Write all rows from the results list

    # print(f"Results saved to {csv_file}")

    final_test_loss, final_psnr, final_ssim = evaluate(model, test_loader, criterion, device, save_sample="test", drop=args.drops, quantize=args.quantize) # constant number of drops
    # final_test_loss, final_psnr, final_ssim = evaluate_consecutive(model, test_loader, criterion, device, drop=args.drops, quantize=args.quantize, consecutive=5) # consecutive drops
    # final_test_loss = evaluate_realistic(model, test_loader, criterion, device, input_drop=args.drops) # random number of drops
    print(f"Final Per-Frame Test Loss for test/evaluation MSE: {final_test_loss:.6f} and PSNR: {final_psnr:.6f} and SSIM: {final_ssim} for tail_len_drops = {args.drops}")
    print("'Global' Dataset-wide PSNR: ", psnr(final_test_loss))
