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
from models import PNC_Autoencoder, PNC16, LRAE_VC_Autoencoder, TestNew, TestNew2, TestNew3
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

    # NOTE: Not necessary but nice to have: Combine and save grouped features
    for prefix, feature_list in grouped_features.items():
        combined_features = np.stack(feature_list, axis=0)  # Combine features
        print("combined_features shape:", combined_features.shape)
        output_file = os.path.join(output_folder, f"{prefix}_combined.npy")
        np.save(output_file, combined_features)
        print(f"Saved combined features for prefix '{prefix}' to '{output_file}'")

def parse_args():
    parser = argparse.ArgumentParser(description="Get features of desired model")
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC16", "LRAE_VC", "TestNew", "TestNew2", "TestNew3"], 
                        help="Model to train")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "PNC":
        model = PNC_Autoencoder().to(device)
        model.load_state_dict(torch.load("PNC_final_no_dropouts.pth")) # NOTE: Load full-features/no random drops model! 
        combined_features_folder = "PNC_combined_features/"
    if args.model == "PNC16":
        model = PNC16().to(device)
        model.load_state_dict(torch.load("PNC16_final_no_dropouts.pth")) 
        combined_features_folder = "PNC16_combined_features/"
    if args.model == "LRAE_VC":
        model = LRAE_VC_Autoencoder().to(device)
        # model.load_state_dict(torch.load("LRAE_VC_final_no_dropouts.pth")) # NOTE: Load full-features/no random drops model! 
        model.load_state_dict(torch.load("LRAE_VC_final_w_random_drops.pth")) # NOTE: Load full-features/no random drops model!
        combined_features_folder = "LRAE_VC_combined_features/"
    if args.model == "TestNew":
        model = TestNew().to(device)
        model.load_state_dict(torch.load("TestNew_final.pth"))
        combined_features_folder = "TestNew_combined_features/"
    if args.model == "TestNew2":
        model = TestNew2().to(device)
        model.load_state_dict(torch.load("TestNew2_final.pth"))
        combined_features_folder = "TestNew2_combined_features/"
    if args.model == "TestNew3":
        model = TestNew3().to(device)
        model.load_state_dict(torch.load("TestNew3_final_no_dropouts.pth"))
        combined_features_folder = "TestNew3_combined_features/"

    
    img_height, img_width = 224, 224  # NOTE: Dependent on autoencoder architecture!!!
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])    # Data loading
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"
    dataset = ImageDataset(path, path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    process_and_save_features(model, data_loader, combined_features_folder, device)
