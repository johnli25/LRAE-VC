import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from models import PNC_Autoencoder, PNC_Autoencoder_NoTail, LRAE_VC_Autoencoder
import argparse
    
# Dataset class for loading images and ground truths
class ImageDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        gt_path = os.path.join(self.gt_dir, self.img_names[idx])

        image = Image.open(img_path).convert("RGB")
        ground_truth = Image.open(gt_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        return image, ground_truth, self.img_names[idx]
    

# Save features to a file
def save_encoder_features(model, dataloader, output_dir, device):
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, _, filenames in dataloader:
            inputs = inputs.to(device)
            features = model.encode(inputs)  # Extract features using encoder
            for i in range(inputs.size(0)):
                feature_array = features[i].cpu().numpy()  # Convert to numpy
                feature_filename = os.path.join(output_dir, f"{filenames[i]}.npy")
                np.save(feature_filename, feature_array)  # Save as .npy file
                print(f"Saved features for {filenames[i]} to {feature_filename}")

# COMBINES FRAMES FROM SAME VIDEO INTO ONE TENSOR
def group_and_combine_features(folder_path, output_folder):
    # Dictionary to store grouped arrays based on prefixes
    grouped_features = defaultdict(list)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".npy"):
            # Extract the prefix (everything before the last underscore)
            prefix = "_".join(filename.split("_")[:-1])
            file_path = os.path.join(folder_path, filename)

            # Load the feature array and add it to the group
            feature_array = np.load(file_path)
            grouped_features[prefix].append(feature_array)

    # Combine arrays for each group and save them
    os.makedirs(output_folder, exist_ok=True)
    for prefix, feature_list in grouped_features.items():
        # Stack arrays along a new axis (e.g., batch axis)
        combined_features = np.stack(feature_list, axis=0)

        # Save the combined array to a new file
        output_file = os.path.join(output_folder, f"{prefix}_combined.npy")
        print(combined_features.shape)
        np.save(output_file, combined_features)
        print(f"Saved combined features for prefix '{prefix}' to '{output_file}'")


    
def process_and_save_features(model, dataloader, output_folder, device):
    """
    Extracts encoder features, groups them by video, and saves combined tensors.
    Args:
        model: The neural network model with an encoder.
        dataloader: DataLoader providing inputs and filenames.
        output_folder: Directory to save combined features.
        device: Device to perform computation on (e.g., 'cuda' or 'cpu').
    """
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_folder, exist_ok=True)

    grouped_features = defaultdict(list)  # Dictionary to store grouped features

    with torch.no_grad():
        for inputs, _, filenames in dataloader:
            inputs = inputs.to(device)
            features = model.encode(inputs)  # Extract features using encoder

            # Group features by video prefix
            for i in range(inputs.size(0)):
                prefix = "_".join(filenames[i].split("_")[:-1])
                feature_array = features[i].cpu().numpy()
                grouped_features[prefix].append(feature_array)

    # Combine and save grouped features
    for prefix, feature_list in grouped_features.items():
        combined_features = np.stack(feature_list, axis=0)  # Combine features
        output_file = os.path.join(output_folder, f"{prefix}_combined.npy")
        np.save(output_file, combined_features)
        print(f"Saved combined features for prefix '{prefix}' to '{output_file}'")

def parse_args():
    parser = argparse.ArgumentParser(description="Get features of desired model")
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_NoTail", "LRAE_VC"], 
                        help="Model to train")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # COMMENT OUT THESE TWO LINES THE SECOND TIME (See Note at Bottom)
    if args.model == "PNC":
        model = PNC_Autoencoder().to(device)
        model.load_state_dict(torch.load("PNC_final_no_dropouts.pth")) # NOTE: Load full-features/no random drops model! 
        combined_features_folder = "PNC_combined_features/"
    if args.model == "LRAE_VC":
        model = LRAE_VC_Autoencoder().to(device)
        # model.load_state_dict(torch.load("LRAE_VC_final_no_dropouts.pth")) # NOTE: Load full-features/no random drops model! 
        model.load_state_dict(torch.load("LRAE_VC_final_w_random_drops.pth")) # NOTE: Load full-features/no random drops model!
        combined_features_folder = "LRAE_VC_combined_features/"
    if args.model == "PNC_NoTail": # unsure if this is necessary or we just need the first two
        model = PNC_Autoencoder_NoTail().to(device)
        model.load_state_dict(torch.load("PNC_NoTail_final_w_random_drops")) # NOTE: Load full-features/no random drops model! 
        combined_features_folder = "PNC_NoTail_combined_features/"
    
    img_height, img_width = 224, 224  # Dependent on autoencoder architecture
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])    # Data loading
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"
    dataset = ImageDataset(path, path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    process_and_save_features(model, data_loader, combined_features_folder, device)
