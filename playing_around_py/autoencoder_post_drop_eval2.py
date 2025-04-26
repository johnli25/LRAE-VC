import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

from models import (
    PNC_Autoencoder, PNC_256Unet_Autoencoder, PNC16,
    TestNew, TestNew2, TestNew3, FrameSequenceLSTM
)


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



# --------------------------------------------------
# Load Pretrained LSTM Models for Feature Imputation
# --------------------------------------------------
lstm_models = {}
input_dim = 1024
hidden_dim = 128
output_dim = 1024
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for filepath in os.listdir("features_num_directory"):
    if "final" in filepath:
        feature_idx = int(filepath.split("_")[1])
        updated_file_path = os.path.join("features_num_directory", filepath)
        model_instance = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
        state_dict = torch.load(updated_file_path, map_location=device)
        model_instance.load_state_dict(state_dict)
        model_instance.eval()  # Set to eval mode
        lstm_models[feature_idx] = model_instance


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

        # Extract video_id and frame_num from file naming convention
        splits = img_name.split("_")
        video_id = "_".join(splits[:-1])
        frame_num = int(splits[-1].split(".")[0])

        return image, img_name, video_id, frame_num


def evaluate_loss_and_fill(ae_model, dataloader, criterion, device, loss_percent):
    """
    Evaluates the model by:
      1) Encoding each input image to get latent.
      2) Zeroing out 'loss_percent' of the latent channels.
      3) Imputing those zeroed channels via the pre-loaded LSTM models.
      4) Decoding the (partly) imputed latent to get reconstructed outputs.
      5) Measuring MSE loss against the original inputs.
    """
    ae_model.eval()
    total_loss = 0.0
    num_samples = 0
    video_latent_dict = defaultdict(dict)

    # Save reconstructed images
    output_path = "LSTM_output_test_imgs_post_train2/"
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path)

    with torch.no_grad():
        # --------------------------------------------------
        # PASS ONE: Compute latent representations (some channels zeroed)
        # --------------------------------------------------
        for inputs, img_name, video_id, frame_num in dataloader:
            img_name, video_id, frame_num = img_name[0], video_id[0], frame_num.item()
            inputs = inputs.to(device)  # shape: [1, 3, H, W]
            
            latent_full = ae_model.encode(inputs)  # shape: [1, C, H_lat, W_lat]

            # Randomly zero out some features
            _, C, H_lat, W_lat = latent_full.shape
            num_zeroed = int((loss_percent / 100.0) * C)
            zero_indices = np.random.choice(C, num_zeroed, replace=False) if num_zeroed > 0 else []

            latent_mod = latent_full.clone()
            for ch_idx in zero_indices:
                latent_mod[0, ch_idx, :, :] = 0

            video_latent_dict[video_id][frame_num] = latent_mod

        # --------------------------------------------------
        # PASS TWO: Impute zeroed features using LSTMs and reconstruct frames
        # --------------------------------------------------
        for inputs, img_name, video_id, frame_num in dataloader:
            img_name, video_id, frame_num = img_name[0], video_id[0], frame_num.item()
            inputs = inputs.to(device)

            frames_dict = video_latent_dict[video_id]
            sorted_frames_nums = sorted(frames_dict.keys())
            seq_len = len(sorted_frames_nums)

            video_latents = []
            for fn in sorted_frames_nums:
                video_latents.append(frames_dict[fn][0])
            video_latents = torch.stack(video_latents, dim=0)  # [seq_len, C, H_lat, W_lat]

            target_index = sorted_frames_nums.index(frame_num)

            # Impute missing channels using LSTM
            _, features_dim, H_lat, W_lat = video_latents.shape
            for feature_idx in range(features_dim):
                if torch.all(video_latents[target_index, feature_idx, :, :] == 0):
                    feature_sequence = video_latents[:, feature_idx, :, :].clone()
                    lstm_feature_model = lstm_models[feature_idx].to(device)

                    predicted_feature = lstm_feature_model(feature_sequence.unsqueeze(0))
                    video_latents[target_index, feature_idx, :, :] = predicted_feature.squeeze(0)[target_index].unsqueeze(0)

            outputs = ae_model.decode(video_latents[target_index].unsqueeze(0))
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
            num_samples += 1

            # Save the reconstructed output images
            output_np = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            plt.imsave(os.path.join(output_path, img_name), output_np)
            # print(f"Saved output image: {img_name} to {output_path}")

    return total_loss / num_samples


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True,
                            choices=["PNC", "PNC_256U", "PNC16", "TestNew",
                                     "TestNew2", "TestNew3"],
                            help="Choose the model architecture")
        parser.add_argument("--path", type=str, required=True,
                            help="Path to the model .pth checkpoint")
        parser.add_argument("--loss_percent", type=int, default=10,
                            help="What percentage of latent channels to zero out during evaluation?")
        return parser.parse_args()

    args = parse_args()

    # Hyperparams
    batch_size = 1
    img_height, img_width = 224, 224
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(path, transform=transform)
    
    test_indices = [
        i for i in range(len(dataset))
        if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
    ]
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "PNC":
        model = PNC_Autoencoder().to(device)
    elif args.model == "PNC_256U":
        model = PNC_256Unet_Autoencoder().to(device)
    elif args.model == "PNC16":
        model = PNC16().to(device)
    elif args.model == "TestNew":
        model = TestNew().to(device)
    elif args.model == "TestNew2":
        model = TestNew2().to(device)
    elif args.model == "TestNew3":
        model = TestNew3().to(device)

    print(f"Loading model weights from {args.path}")
    model.load_state_dict(torch.load(args.path, map_location=device))
    criterion = nn.MSELoss()

    # Run Evaluation
    test_loss = evaluate_loss_and_fill(model, test_loader, criterion, device, args.loss_percent)
    print(f"Final Test Loss: {test_loss:.4f}")


