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
import argparse

# TODO: 
# ADD FINAL LRAE-VC Architecture
    
class LRAE_VC_Autoencoder(nn.Module):
    def __init__(self):
        super(LRAE_VC_Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # (3, 224, 224) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (32, 112, 112) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 56, 56) -> (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 28, 28) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (64, 56, 56) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (32, 112, 112) -> (16, 224, 224)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        self.final_layer = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # (16, 224, 224) -> (3, 224, 224)

        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)  # (32, 112, 112)
        x2 = self.encoder2(x1)  # (64, 56, 56)
        x3 = self.encoder3(x2)  # (128, 28, 28)

        # Decoder
        y1 = self.decoder1(x3)  # (64, 56, 56)
        y1 = y1 + x2  # Skip connection

        y2 = self.decoder2(y1)  # (32, 112, 112)
        y2 = y2 + x1  # Skip connection

        y3 = self.decoder3(y2)  # (16, 224, 224)

        y4 = self.final_layer(self.dropout(y3))  # (3, 224, 224)
        y5 = self.sigmoid(y4)  # Normalize output to [0, 1]
        return y5
    
    def encode(self, x):
        # Encoder
        x1 = self.encoder1(x)  # (32, 112, 112)
        x2 = self.encoder2(x1)  # (64, 56, 56)
        x3 = self.encoder3(x2)  # (128, 28, 28)
        return x3
    
class PNC_Autoencoder(nn.Module):
    def __init__(self):
        super(PNC_Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (16, 32, 32)
        self.encoder2 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)  # (16, 32, 32) -> (10, 32, 32)

        # Decoder
        self.decoder1 = nn.ConvTranspose2d(10, 64, kernel_size=9, stride=7, padding=4, output_padding=6)  # (10, 32, 32) -> (64, 224, 224)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (64, 224, 224) -> (3, 224, 224)

        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For output normalization in range [0, 1]

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.encoder1(x))  # (16, 32, 32)
        # print(f"Shape after encoder1: {x1.shape}")
        x2 = self.relu(self.encoder2(x1))  # (10, 32, 32)
        # print(f"Shape after encoder2: {x2.shape}")

        # Decoder
        y1 = self.relu(self.decoder1(x2))  # (64, 224, 224)
        # print(f"Shape after decoder1: {y1.shape}")

        y2 = self.relu(self.decoder2(y1))  # (64, 224, 224)
        # print(f"Shape after decoder2: {y2.shape}")
        y2 = y2 + y1 # Skip connection

        y3 = self.relu(self.decoder3(y2))  # (64, 224, 224)
        # print(f"Shape after decoder3: {y3.shape}")

        y4 = self.relu(self.decoder3(y3))  # (64, 224, 224)
        # print(f"Shape after decoder3 (second time): {y4.shape}")
        y4 = y4 + y3  # Skip connection

        y5 = self.final_layer(y4)  # (3, 224, 224)
        # print(f"Shape after final_layer: {y5.shape}")
        y5 = torch.clamp(y5, min=0, max=1)  # Ensure output is in [0, 1] range
        return y5
    
    def encode(self, x):
        # Encoder
        x1 = self.relu(self.encoder1(x))  # (16, 32, 32)
        # print(f"Shape after encoder1: {x1.shape}")
        x2 = self.relu(self.encoder2(x1))  # (10, 32, 32)
        # print(f"Shape after encoder2: {x2.shape}")
        return x2
    
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

if __name__ == "__main__":
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = PNC_Autoencoder().to(device)
    #model.load_state_dict(torch.load("PNC_best_validation.pth"))
    
    img_height, img_width = 224, 224  # Dependent on autoencoder architecture
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])    # Data loading
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"
    dataset = ImageDataset(path, path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Save encoder features for test set
    features_output_dir = "PNC_encoder_features/"
    encoder_features_folder = "PNC_encoder_features/"
    combined_features_folder = "PNC_combined_features/"

    # If this doesnt work, run just save_encoder_features and just group_and_combine_features (might have to run group_and_combine_features without cuda)
    #save_encoder_features(model, data_loader, features_output_dir, device) # Saving latent features for each frame
    group_and_combine_features(encoder_features_folder, combined_features_folder) 