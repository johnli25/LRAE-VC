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


video_features_lookup = {}



# Preload + cache LSTM models to avoid reloading them in the loop
lstm_models = {}
input_dim = 1024      # e.g., flattened spatial dims from [32, 32]
hidden_dim = 128
output_dim = 1024     # same as input_dim typically
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for filepath in os.listdir("features_num_directory"):
    if "final" in filepath:
        feature_idx = int(filepath.split("_")[1])
        updated_file_path = os.path.join("features_num_directory", filepath)
        # Create a new instance of your LSTM model
        model_instance = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
        # Load the state dict into the model instance
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
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Use the same image for both input and ground truth
        return image, image, self.img_names[idx]  # (image, same_image_as_ground_truth, img filename)
    

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_256U", "PNC16", "TestNew", "TestNew2", "TestNew3", "PNC_NoTail", "PNC_with_classification", "LRAE_VC"], 
                            help="Model to train")
        return parser.parse_args()

    args = parse_args()



    ## Hyperparameters
    batch_size = 1
    img_height, img_width = 224, 224 # NOTE: Dependent on autoencoder architecture!!
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/" # NOTE: already resized to 224x224 (so not really adaptable), but faster
    # path = "UCF_uncompressed_video_img_frames" # NOTE: more adaptable to different img dimensions because it's the original, BUT SLOWER!

    # Data loading
    transform = transforms.Compose([
        # transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(path, transform=transform)
    test_indices = [
        i for i in range(len(dataset))
        if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
    ]
    train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

    test_dataset = Subset(dataset, test_indices)

    print(f"Full dataset size: {len(dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # apply data augmentation to the train data       
    test_dataset.dataset.transform = transform

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_tail_length = None
    video_features_dir = None
    if args.model == "PNC":
        model = PNC_Autoencoder().to(device)
        max_tail_length = 10 # NOTE: true PNC

    if args.model == "PNC_256U":
        model = PNC_256Unet_Autoencoder().to(device)

    if args.model == "PNC16":
        model = PNC16().to(device)
        video_features_dir = "PNC16_combined_features/"
        model.load_state_dict(torch.load("PNC16_post_drop_fill_final.pth"))

    if args.model == "TestNew":
        model = TestNew().to(device)

    if args.model == "TestNew2":
        model = TestNew2().to(device)

    if args.model == "TestNew3":
        model = TestNew3().to(device)
    

    # NOTE: Preload + cache Video features:
    for features_file in os.listdir(video_features_dir):
        video_id = "_".join(features_file.split("_")[:-1])
        video_features_lookup[video_id] = np.load(os.path.join(video_features_dir, features_file))


    # Save images generated by DECODER ONLY! 
    output_path = "LSTM_output_test_imgs_post_train/"
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path)

    criterion = nn.MSELoss()
    loss_percent = 91
    test_loss = 0.0
    with torch.no_grad():
       for batch, (inputs, targets, filenames) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            filename = filenames[0] # NOTE: filename is a tuple containing strings of size batch_size (which is set to 1)
            # print("Filename:", filename)     
            video_id = "_".join(filename.split("_")[:-1])   
            frame_num = int(filename.split("_")[-1].split(".")[0]) - 1 # removes ".jpg" and everything before the frame number (action + video num). ALSO, frame num is 1-indexed.
            video_features = torch.tensor(video_features_lookup[video_id]).to(device)
            # print(f"filename: {filename}, video_features.shape: {video_features.shape}")

            seq_len, feature_dim = video_features.shape[0], video_features.shape[1]
            num_zeroed_features = int((loss_percent / 100.0) * feature_dim)
            zero_indices = np.random.choice(feature_dim, num_zeroed_features, replace=False)
            # zero_indices = np.random.choice(feature_dim, 0, replace=False)
            print("loss_percent and zero_indices", loss_percent, zero_indices)
            latent = model.encode(inputs)
            latent_mod = latent.clone()

            # print("latent shape", latent_mod.shape)
            for feature_idx in zero_indices:
                # zero out the latent feature for the current frame
                video_features[frame_num, feature_idx, :, :] = 0 # NOTE: b/c feature_idx is 1-indexed
                # pass this into lstm_model 
                dropped_feature_sequence = video_features[:, feature_idx, :, :].clone() # shape: (seq_len, height, width)
                # print("dropped feature sequence:", dropped_feature_sequence.shape) 
                lstm_feature_model = lstm_models[feature_idx]
                with torch.no_grad():
                    predicted_feature = lstm_feature_model(dropped_feature_sequence.unsqueeze(0)) # shape: (1, seq_len, height, width)
                    # print("predicted_feature shape:", predicted_feature.shape) 

                latent_mod[:, feature_idx, :, :] = predicted_feature.squeeze(0)[frame_num].unsqueeze(0)

            outputs = model.decode(latent_mod)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0) # where inputs.size(0) is the BATCH size

            for j in range(inputs.size(0)):
                output_np = outputs[j].permute(1, 2, 0).cpu().numpy()  # (image_h, image_w, 3)
                plt.imsave(os.path.join(output_path, filenames[j]), output_np)

    print(f"Test Loss: {test_loss / len(test_loader.dataset)}")
