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
from models import PNC_Autoencoder, PNC_256Unet_Autoencoder, PNC16, TestNew, TestNew2, TestNew3, PNC_with_classification, FrameSequenceLSTM
import random
from collections import defaultdict


# NOTE: uncomment below if you're using UCF Sports Action 
class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

test_img_names = {
    "diving_7", "diving_8", "golf_front_7", "golf_front_8", "kick_front_8", "kick_front_9",
    "lifting_5", "lifting_6", "riding_horse_8", "riding_horse_9", "running_7", "running_8",
    "running_9", "skating_8", "skating_9", "swing_bench_7", "swing_bench_8", "swing_bench_9"
}

# NOTE: uncomment below if you're using UCF101


# This GLOBAL dictionary maps: 
#   video_id -> dict of frame_num -> latent of shape [1, C, H_lat, W_lat]
# i.e. for each frame in the video, we store the *possibly zeroed* latent.
video_latent_dict = defaultdict(dict)


# Preload + cache LSTM models to avoid reloading them in the loop
lstm_models = {}
input_dim = 1024
hidden_dim = 128
output_dim = 1024
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for filepath in os.listdir("features_num_directory"):
    if "final" in filepath:
        feature_idx = int(filepath.split("_")[1])
        updated_file_path = os.path.join("features_num_directory", filepath)
        
        # 1) Create a new instance of your LSTM model (on CPU for the moment)
        model_instance = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers)
        
        # 2) Load the state dict
        state_dict = torch.load(updated_file_path, map_location=device)
        model_instance.load_state_dict(state_dict)
        
        # 3) Put in eval mode
        model_instance.eval()
        
        # 4) Move it to the GPU (or CPU if no GPUs)
        model_instance = model_instance.to(device)
        
        # 5) Wrap it in DataParallel if you want multi-GPU
        if torch.cuda.device_count() > 1:
            model_instance = nn.DataParallel(model_instance)
        
        lstm_models[feature_idx] = model_instance # 6) store in the lstm_models dictionary



class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Derive video_id and frame_num from file naming convention
        splits = img_name.split("_")
        video_id = "_".join(splits[:-1])
        frame_str = splits[-1].split(".")[0]  # e.g. "037"
        frame_num = int(frame_str)            # now a Python int, not a tensor
        # print(f"img_name: {img_name}, video_id: {video_id}, frame_num: {frame_num}")
        return image, img_name, video_id, frame_num
    

def plot_train_val_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('train_val_loss_curve.png', dpi=300)
    plt.show()


def train(ae_model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name):
    """
    1) Pass One: 
        - Build (and update) a dictionary of latents for each video_id.
        - randomly zero-out channels/features in these latents.
    2) Pass Two:
        - For each frame in the dataloader, fetch the entire video's latents,
        impute missing channels for the target frame using LSTM, decode that frame, compute loss, backprop.
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    loss_percentages = [0, 10, 20, 30, 40, 50, 60 , 70 , 80, 90]

    for epoch in range(num_epochs):
        ae_model.eval()  
        # PASS ONE: Fill the dictionary latents (some channels zeroed)
        with torch.no_grad():
            for (inputs, img_name, video_id, frame_num) in train_loader:
                img_name, video_id, frame_num = img_name[0], video_id[0], frame_num[0].item()  # Get the first name in the batch (which in most cases is of size 1 anyway)
                # print(f"PASS 1: img_name: {img_name}, video_id: {video_id}, frame_num: {frame_num}, {type(frame_num.item())}")
                inputs = inputs.to(device) # inputs shape: [1, 3, H, W]
                if isinstance(ae_model, nn.DataParallel):
                    latent_full = ae_model.module.encode(inputs)  # shape: [1, C, H_lat, W_lat]
                else:
                    latent_full = ae_model.encode(inputs)

                # random zero out some features
                _, C, H_lat, W_lat = latent_full.shape
                loss_percent = np.random.choice(loss_percentages)
                num_zeroed = int((0 / 100.0) * C)
                zero_indices = np.random.choice(C, num_zeroed, replace=False) if num_zeroed > 0 else []

                latent_mod = latent_full.clone()
                for ch_idx in zero_indices:
                    latent_mod[0, ch_idx, :, :] = 0

                # Store in the dictionary. (We assume each frame_num is unique within that video)
                video_latent_dict[video_id][frame_num] = latent_mod # NOTE: this is easier than structuring video_latent_dict as {video_id : video_tensor_of_shape[seq_len, C, H_lat, W_lat]} b/c it's easier to process individual frames randomly this way

        # PASS TWO: Train the model using the possibly zeroed latents
        epoch_loss, num_samples = 0.0, 0
        ae_model.train()  # Set the model to training mode

        for inputs, img_name, video_id, frame_num in train_loader:
            img_name, video_id, frame_num = img_name[0], video_id[0], frame_num[0].item()  # Get the first name in the batch (which in most cases is of size 1 anyway)
            # print(f"PASS 2: img_name: {img_name}, video_id: {video_id}, frame_num: {frame_num}")
            inputs = inputs.to(device)

            frames_dict = video_latent_dict[video_id]
            # sort by frame num to produce shape: [seq_len, C, H_lat, W_lat]
            sorted_frames_nums = sorted(frames_dict.keys())
            seq_len = len(sorted_frames_nums)

            video_latents = []
            for fn in sorted_frames_nums: # 'fn' is shorthand for frame number NOTE: DIFFERENT FROM frame_num in dataloader! 
                # frames_dict[fn] is shape [1, C, H, W], 
                # so frames_dict[fn][0] extracts the [C, H, W] for easier stacking
                video_latents.append(frames_dict[fn][0])
            video_latents = torch.stack(video_latents, dim=0) # [seq_len, C, H_lat, W_lat]  

            _, features_dim, H_lat, W_lat = video_latents.shape
            for feature_idx in range(features_dim):
                # check if feature channel is zeroed
                if torch.all(video_latents[frame_num - 1, feature_idx, :, :] == 0): # torch.all is VECTORIZED (not manual for loop) so it's actually fast
                    # impute using LSTM
                    feature_videosequence = video_latents[:, feature_idx, :, :].clone() # shape [seq_len, H_lat, W_lat]
                    lstm_feature_model = lstm_models[feature_idx]
                    with torch.no_grad():
                        predicted_feature = lstm_feature_model(feature_videosequence.unsqueeze(0)) # shape [1, seq_len, H_lat, W_lat]

                    # replace the zeroed feature with the predicted feature
                    video_latents[frame_num - 1, feature_idx, :, :] = predicted_feature.squeeze(0)[frame_num - 1].unsqueeze(0)

            # Now Decode the modified latents
            optimizer.zero_grad()   
            if isinstance(ae_model, nn.DataParallel):
                outputs = ae_model.module.decode(video_latents[frame_num - 1].unsqueeze(0))  # shape: [1, 3, H, W]
            else:
                outputs = ae_model.decode(video_latents[frame_num - 1].unsqueeze(0))
            loss = criterion(outputs, inputs[0])  # Compare with the original input image
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_samples += 1

        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            if isinstance(ae_model, nn.DataParallel):
                torch.save(ae_model.module.state_dict(), f"{model_name}_post_drop_fill_final2.pth") # save "final" model every iteration so we can resume training if needed
            else:
                torch.save(ae_model.state_dict(), f"{model_name}_post_drop_fill_final2.pth")

        # Compute average loss for the epoch
        epoch_loss /= num_samples
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Skip Validation for now!

    # Final evaluation on test set
    test_loss = evaluate_normal(ae_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")


def evaluate_normal(model, dataloader, criterion, device): # NOTE: without any zeroing out due to packet drops/loss (0% percent_loss)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, img_name, video_id, frame_num in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Must ensure your model.forward() => encode+decode
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)



# Main driver function
if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True,
                            choices=["PNC", "PNC_256U", "PNC16", "TestNew",
                                     "TestNew2", "TestNew3", "PNC_NoTail",
                                     "PNC_with_classification", "LRAE_VC"],
                            help="Choose your model")
        parser.add_argument("--checkpoint", type=str, default=None,
                            help="Path to a .pth model checkpoint to load.")
        parser.add_argument("--train_from_scratch", action="store_true", # NOTE: you don't have to pass that flag into start training from scratch! 
                            help="If set, always train from scratch (ignore checkpoint if provided).")
        parser.add_argument("--num_epochs", type=int, default=40, help="Number of epochs to train.")
        
        return parser.parse_args()

    args = parse_args()

    # Hyperparams
    batch_size = 1
    learning_rate = 1e-3
    img_height, img_width = 224, 224
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(path, transform=transform)

    # Example: filtering test images
    test_indices = [
        i for i in range(len(dataset))
        if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
    ]
    train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

    np.random.shuffle(train_val_indices)
    val_fraction = 0.0 # NOTE: val dataset is currently NOT used (denoted by setting to 0)! 
    val_size = int(val_fraction * len(train_val_indices))
    train_indices = train_val_indices[val_size:]
    val_indices = train_val_indices[:val_size]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "PNC":
        model = PNC_Autoencoder()
    elif args.model == "PNC_256U":
        model = PNC_256Unet_Autoencoder()
    elif args.model == "PNC16":
        model = PNC16()
    elif args.model == "TestNew":
        model = TestNew()
    elif args.model == "TestNew2":
        model = TestNew2()
    elif args.model == "TestNew3":
        model = TestNew3()

    model = model.to(device)

    ### if you want to enable DataParallel (on multiple GPUs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    ### Otherwise ignore above

    if args.checkpoint is not None and not args.train_from_scratch:
        print(f"Loading model weights from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, None, test_loader, # NOTE: setting val_loader to None now to keep things simple
        criterion, optimizer, device, args.num_epochs, model_name=args.model)
