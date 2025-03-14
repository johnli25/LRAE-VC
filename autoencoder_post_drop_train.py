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

# NOTE: uncomment below if you're using UCF101


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

# Dataset class for loading images and ground truths
class VideoDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

        self.video_dict = {}
        for filename in self.img_names:
            # assume the video name is everything before the last underscore
            # e.g., "diving_7.jpg" -> video id: "diving"
            video_id = "_".join(filename.split("_")[:-1])
            if video_id not in self.video_dict:
                self.video_dict[video_id] = []
            self.video_dict[video_id].append(filename)
        
        # sort the frames within each video by frame number (assumes the last part of the filename before extension is the frame number)
        for video_id, file_list in self.video_dict.items():
            self.video_dict[video_id] = sorted(file_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # create a list of video IDs for indexing
        self.video_ids = list(self.video_dict.keys())
        print(self.video_ids)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frame_files = self.video_dict[video_id]
        frames = []
        for file in frame_files:
            img_path = os.path.join(self.img_dir, file)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        video_tensor = torch.stack(frames, dim=0)  # (num_frames AKA seq_len, 3, img_height, img_width)
        return video_tensor, video_id # (video_tensor, video filename)
    

class DummyVideoDataset(Dataset):
    def __init__(self, img_dir, seq_len=32, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing image files.
            seq_len (int): Number of frames per video sequence.
            transform (callable, optional): Transform to apply to each image.
        """
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        # List all image files (adjust extensions as needed)
        self.img_names = os.listdir(img_dir)
        
        self.video_dict = {}
        for filename in self.img_names:
            # assume the video name is everything before the last underscore
            # e.g., "diving_7.jpg" -> video id: "diving"
            video_id = "_".join(filename.split("_")[:-1])
            if video_id not in self.video_dict:
                self.video_dict[video_id] = []
            self.video_dict[video_id].append(filename)
        
        self.video_ids = list(self.video_dict.keys())
        print(self.video_ids)

    def __len__(self):
        # You can define the length arbitrarily.
        # Here, we return the number of sequences we can get from the total images.
        # For example, if there are 320 images and seq_len is 32, then we have 10 samples.
        return len(self.img_names) // self.seq_len

    def __getitem__(self, idx):
        # Instead of grouping by video, we randomly sample `seq_len` images from the entire image pool.
        sampled_names = np.random.choice(self.img_names, self.seq_len, replace=False)
        frames = []
        for file in sampled_names:
            img_path = os.path.join(self.img_dir, file)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        # Stack frames into a video tensor: shape [seq_len, C, H, W]
        video_tensor = torch.stack(frames, dim=0)
        video_id = f"dummy_video_{idx}"  # Dummy video ID
        return video_tensor, video_id


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

    plt.savefig('train_val_loss_curve_post_drop_fill.png', dpi=300)
    plt.show()

def train(ae_model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    loss_percentages=[0, 10, 20, 30, 40, 50, 60, 70, 80]

    for epoch in range(num_epochs):
        # Train the model
        ae_model.train()
        train_loss, total_frames = 0.0, 0

        for inputs, targets, filename in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            filename = filename[0] # NOTE: filename is a tuple containing strings of size batch_size (which is set to 1)
            video_id = "_".join(filename.split("_")[:-1])    
            frame_num = int(filename.split("_")[-1].split(".")[0]) - 1 # removes ".jpg" and everything before the frame number (action + video num). ALSO, frame num is 1-indexed.
            video_features = torch.tensor(video_features_lookup[video_id]).to(device)
            print(f"filename: {filename}, video_features.shape: {video_features.shape}")

            seq_len, feature_dim = video_features.shape[0], video_features.shape[1]
            loss_percent = np.random.choice(loss_percentages)
            num_zeroed_features = int((loss_percent / 100.0) * feature_dim)
            zero_indices = np.random.choice(feature_dim, num_zeroed_features, replace=False)

            # print("loss_percent and zero_indices", loss_percent, zero_indices)
            latent = ae_model.encode(inputs)
            latent_mod = latent.clone()

            optimizer.zero_grad() # clear/zero out gradients 

            # print("latent shape", latent_mod.shape)
            ##### DEBUG CHECK if any latent features have all 0s #####
            for feature_idx in range(feature_dim):
                if torch.all(latent_mod[:, feature_idx, :, :] == 0.0):
                    print("Sanity check feature shape: ", latent_mod[:, feature_idx, :, :].shape)
                    print(f"Feature {feature_idx} is all zeros in latent_mod for {filename}")
            ##### END DEBUG CHECK 

            for feature_idx in zero_indices:
                # zero out the latent feature for the current frame
                video_features[frame_num, feature_idx, :, :] = 0 
                # pass this into lstm_model 
                dropped_feature_sequence = video_features[:, feature_idx, :, :].clone() # shape: (seq_len, height, width)
                # print("dropped feature sequence:", dropped_feature_sequence.shape) 
                lstm_feature_model = lstm_models[feature_idx]
                with torch.no_grad():
                    predicted_feature = lstm_feature_model(dropped_feature_sequence.unsqueeze(0)) # shape: (1, seq_len, height, width)
                    # print("predicted_feature shape:", predicted_feature.shape) 

                latent_mod[:, feature_idx, :, :] = predicted_feature.squeeze(0)[frame_num].unsqueeze(0)

            outputs = ae_model.decode(latent_mod)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        val_loss = evaluate(model=ae_model, dataloader=val_loader, criterion=criterion, device=device, loss_percent=31)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ae_model.state_dict(), f"{model_name}_post_drop_fill_best_validation.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    plot_train_val_loss(train_losses, val_losses)

    # Save final model
    torch.save(ae_model.state_dict(), f"{model_name}_post_drop_fill_final.pth")

    # Final Test: test_autoencoder()
    test_loss = evaluate(model=ae_model, dataloader=test_loader, criterion=criterion, device=device, loss_percent=31)
    print(f"Final Test Loss: {test_loss:.4f}")


def evaluate(model, dataloader, criterion, device, loss_percent):
    model.eval()
    test_loss = 0
    with torch.no_grad():
       for inputs, targets, filenames in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            filename = filenames[0] # NOTE: filename is a tuple containing strings of size batch_size (which is set to 1)
            video_id = "_".join(filename.split("_")[:-1])    
            frame_num = int(filename.split("_")[-1].split(".")[0]) - 1 # removes ".jpg" and everything before the frame number (action + video num). ALSO, frame num is 1-indexed.
            video_features = torch.tensor(video_features_lookup[video_id]).to(device)
            # print(f"filename: {filename}, video_features.shape: {video_features.shape}")

            seq_len, feature_dim = video_features.shape[0], video_features.shape[1]
            num_zeroed_features = int((loss_percent / 100.0) * feature_dim)
            zero_indices = np.random.choice(feature_dim, num_zeroed_features, replace=False)
            # print("loss_percent and zero_indices", loss_percent, zero_indices)
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

    return test_loss / len(dataloader.dataset) 



if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_256U", "PNC16", "TestNew", "TestNew2", "TestNew3", "PNC_NoTail", "PNC_with_classification", "LRAE_VC"], 
                            help="Model to train")
        return parser.parse_args()

    args = parse_args()



    ## Hyperparameters
    num_epochs = 50
    batch_size = 1 # NOTE: set to 1 because each sample is a full video sequence and videos may have different number of frames/sequence lengths
    learning_rate = 1e-3
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

    # Split train_val_indices into train and validation
    np.random.shuffle(train_val_indices)
    train_size = int(0.9 * len(train_val_indices))
    val_size = len(train_val_indices) - train_size

    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Full dataset size: {len(dataset)}")
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

    if args.model == "TestNew":
        model = TestNew().to(device)

    if args.model == "TestNew2":
        model = TestNew2().to(device)

    if args.model == "TestNew3":
        model = TestNew3().to(device)

    # NOTE: Preload + cache Video features:
    for filename in os.listdir(video_features_dir):
        video_id = "_".join(filename.split("_")[:-1])
        video_features_lookup[video_id] = np.load(os.path.join(video_features_dir, filename))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    
    train(ae_model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, device=device, num_epochs=num_epochs, model_name=args.model)

    # Save images generated by DECODER ONLY! 
    output_path = "LSTM_output_test_imgs_post_train/"
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path)

    model.eval()  # Put the autoencoder in eval mode
    with torch.no_grad():
        for i, (inputs, _, filenames) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)  # Forward pass through autoencoder
            # print("Outputs shape:", outputs.shape)

            # outputs is (batch_size, 3, image_h, image_w)
            for j in range(inputs.size(0)):
                output_np = outputs[j].permute(1, 2, 0).cpu().numpy()  # (image_h, image_w, 3)
                plt.imsave(os.path.join(output_path, filenames[j]), output_np)
