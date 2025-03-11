import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from models import FrameSequenceLSTM
import argparse
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, directory, feature_num):
        """
        Args:
            directory (str): Path to the directory containing the .npy files.
            feature_num (int): Index of the feature to extract from each file.
        """
        # List all files in the directory
        self.file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npy')]
        self.num = feature_num
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index to access a specific file.
        
        Returns:
            torch.tensor: A tensor of shape (sequence_length, 32, 32) representing the selected feature sequence.
        """
        file_path = self.file_paths[idx]
        
        # Load the file (assuming it's in numpy format)
        data = np.load(file_path)  # Expected shape: (X, 10, 32, 32) or similar
        swapped_data = np.swapaxes(data, 0, 1)  # Swap axes to get (10, X, 32, 32)
        # print("file and data shape: ", file_path, swapped_data.shape) # NOTE: this print statement useful for debugging!!

        # Extract the specified feature sequence
        sequence_item = swapped_data[self.num]  
        
        # Convert to a torch tensor
        return torch.tensor(sequence_item, dtype=torch.float32), file_path # shape: [seq_length, 32, 32] b/c converted from [feature_i, seq_length, 32, 32] to [seq_length, 32, 32])


# phase 1: Pre-Training on Complete Sequences
def pretrain_lstm_on_complete_sequences(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss, total_batches = 0, 0
    for batch_idx, (features, filepath) in enumerate(dataloader):
        features = features.to(device) # shape is [batch_size, seq_length, 32, 32] 
        target = features.clone()
        output_seq = model(features)
        loss = criterion(output_seq, target).mean()  # Loss is computed as average over the entire sequence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_batches += 1

    return epoch_loss / total_batches

# phase 2: Fine-Tuning on Sequences with Random Drops (one frame at a time)
def finetune_lstm_on_dropped_latents_single(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss, total_batches = 0, 0
    for batch_idx, (features, filepath) in enumerate(dataloader):
        features = features.to(device)
        batch_size, seq_len, latent_dim_height, latent_dim_width = features.shape 

        drop_index = np.random.randint(seq_len)
        target_latent = features[:, drop_index, :, :].clone()

        input_sequence = features.clone()
        input_sequence[:, drop_index, :, :] = 0

        output_seq = model(input_sequence)

        predicted_latent = output_seq[:, drop_index, :, :]
        print("predicted and target shape", predicted_latent.shape, target_latent.shape)
        loss = criterion(predicted_latent, target_latent) # Loss is computed only on the DROPPED frame
        print("loss: ", loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_batches += 1

    return epoch_loss / total_batches

# phase 2 (alternative): Fine-Tuning on Sequences with Random Drops (multiple frames at a time randome=ly)
def finetune_lstm_on_dropped_latents_multi_random_loss(model, dataloader, criterion, optimizer, device, loss_percentages=[0, 10, 20, 30, 40, 50, 60, 70, 80]):
    model.train()
    epoch_loss = 0
    for batch_idx, (features, filepath) in enumerate(dataloader):
        features = features.to(device)  
        batch_size, seq_len, latent_dim_height, latent_dim_width = features.shape

        loss_percent = np.random.choice(loss_percentages)
        num_zeroed_features = int((loss_percent / 100.0) * seq_len)

        # print(f"Loss %={loss_percent}%; seq_len={seq_len}; num_zeroed_features={num_zeroed_features}")
        if num_zeroed_features == 0: 
            # print("No frames dropped in this batch!")
            # num_zeroed_features = 1 # NOTE-Option 1: ASSUME at least one frame is always dropped (experiment later if this assumption actually valid)!!!
            continue # NOTE-Option 2: OR assume that if no frames dropped, then skip the batch and don't compute loss + backpropagate
    
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)   
        zero_indices = np.random.choice(seq_len, num_zeroed_features, replace=False)
        mask[zero_indices] = True # shape: [seq_len]

        mask_expanded = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, latent_dim_height, latent_dim_width) # `expand_as` converts from [seq_len] to [batch_size, seq_len, 32, 32]
        input_features = features.clone()   
        input_features[mask_expanded] = 0

        # forward pass - execute one step of the model:
        output_seq = model(input_features)
        loss = criterion(output_seq, features)  

        masked_loss = loss * mask_expanded.float()  # Only compute loss on dropped frames (optional)
        num_dropped_elements = mask_expanded.float().sum()
        batch_loss = masked_loss.sum() / num_dropped_elements
        # print(f"TRAINING: Loss info: original loss shape={loss.shape}; original loss sum={loss.sum()}; masked_loss={masked_loss.sum()}; batch loss={batch_loss.item()}, num_dropped_elements={num_dropped_elements.item()}")
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()    

        epoch_loss += batch_loss.item()

    return epoch_loss
    

def eval_model(model, dataloader, criterion, device, loss_percentage): 
    """
    Evaluate the model under a FIXED packet loss rate (GRACE-style evaluation) and compute average loss 
    only on the dropped frames.

    Args:
        model (nn.Module): The trained LSTM model.
        dataloader (DataLoader): DataLoader containing test/validation sequences.
        criterion (nn.Module): Loss function (e.g., nn.MSELoss(reduction='none')).
        device (torch.device): 'cuda' or 'cpu'.
        loss_percentage (float): The fixed packet loss rate (e.g., 10, 20, 30, etc.).

    Returns:
        float: Average loss over only the dropped (masked) elements across the dataset.
    """
    model.eval()
    epoch_loss, total_dropped_elements = 0, 0
    with torch.no_grad():
        for (features, filepath) in dataloader:
            features = features.to(device)  # Shape: [batch_size, seq_length, height, width]
            batch_size, sequence_length, latent_dim_height, latent_dim_width = features.shape

            num_zeroed_frames = int((loss_percentage / 100.0) * sequence_length)
            if num_zeroed_frames == 0:
                continue  # Skip batches where no loss occurs (SOLE PURPOSE: prevents division by zero)

            mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
            zero_indices = np.random.choice(sequence_length, num_zeroed_frames, replace=False)
            mask[zero_indices] = True

            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(batch_size, sequence_length, latent_dim_height, latent_dim_width)

            input_features = features.clone()
            input_features[mask_expanded] = 0  # Apply packet loss mask

            # Forward pass
            output_seq = model(input_features)

            # Compute loss
            loss = criterion(output_seq, features)  # MSELoss (shape: [batch_size, seq_length, height, width])

            # Apply mask to compute loss only on dropped frames
            masked_loss = loss * mask_expanded.float()

            # Sum loss only over dropped elements
            num_dropped = mask_expanded.float().sum()

            batch_loss = masked_loss.sum() / num_dropped

            # print(f"VALIDATION: Loss info: original loss shape={loss.shape}; original loss sum={loss.sum()}; masked_loss={masked_loss.sum()}; batch loss={batch_loss.item()}, num_dropped_elements={num_dropped.item()}")
            if num_dropped == 0:continue  # Skip batch if no frames were dropped

            epoch_loss += batch_loss.item()
            # total_dropped_elements += num_dropped.item()

    return epoch_loss # / total_dropped_elements if total_dropped_elements > 0 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Get features of desired model")
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC16", "LRAE_VC", "TestNew", "TestNew2", "TestNew3"], 
                        help="Model to train")
    return parser.parse_args()


def plot_train_val_loss(train_losses, val_losses, feature_num):
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

    plt.savefig(f'train_val_loss_curve_fill_feature_{feature_num}.png', dpi=300)    
    plt.show()


if __name__ == "__main__": 
    args = parse_args()
    # Configuration
    if args.model == "PNC":
        input_dim = 32 * 32
        output_dim = 32 * 32
        num_features = 10    
        folder_path = "PNC_combined_features"
    if args.model == "PNC16":
        input_dim = 32 * 32
        output_dim = 32 * 32
        num_features = 16
        folder_path = "PNC16_combined_features"
    if args.model == "LRAE_VC":
        input_dim = 28 * 28
        output_dim = 28 * 28
        num_features = 16    
        folder_path = "LRAE_VC_combined_features"
    if args.model == "TestNew":
        input_dim = 48 * 48
        output_dim = 48 * 48
        num_features = 24    
        folder_path = "TestNew_combined_features"
    if args.model == "TestNew2":
        input_dim = 56 * 56
        output_dim = 56 * 56
        num_features = 24    
        folder_path = "TestNew2_combined_features"
    if args.model == "TestNew3":
        input_dim = 38 * 38
        output_dim = 38 * 38
        num_features = 24    
        folder_path = "TestNew3_combined_features"


    # Hyperparameters
    epochs = 120
    batch_size = 1
    hidden_dim = 128  # Tuned hyperparameter for LSTM
    num_layers = 2
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if not os.path.exists("features_num_directory"):
        os.makedirs("features_num_directory")

    for i in range(num_features):
        # TRAINING FOR FEATURE INDEX i
        print(f"Training for Feature {i}")
        model = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
        criterion = nn.MSELoss(reduction='none') 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        dataset = CustomDataset(folder_path, i)

        ##### NOTE: DEBUG #####
        # print(f"Total number of samples in the dataset: {len(dataset)}")
        # print
        # for i in range(min(3, len(dataset))):  # Print the first 3 samples or less if the dataset is smaller
        #     sample, filepath = dataset[i]
        #     print(f"Sample {i} shape: {sample.shape}")
        ##### END DEBUG #####
        
        # Shuffle dataset indices and create splits: 82% train, 8% validation, 10% test
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        train_end = int(0.82 * len(dataset))
        val_end = int(0.90 * len(dataset))
        train_indices = all_indices[:train_end]
        val_indices = all_indices[train_end:val_end]
        test_indices = all_indices[val_end:]

        # Subsets for train, validation, and test
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        min_val_loss = float('inf')

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # TRAINING PHASE
            pretrain_loss = pretrain_lstm_on_complete_sequences(model, train_dataloader, criterion, optimizer, device)
            print(f"[Feature {i}] Epoch {epoch}, Pretrain Loss (no feature packet drops): {pretrain_loss:.4f}")
            train_loss = finetune_lstm_on_dropped_latents_multi_random_loss(model, train_dataloader, criterion, optimizer, device)
            print(f"[Feature {i}] Epoch {epoch}, Training Loss: {train_loss:.4f}")
            train_losses.append(train_loss)
            
            # VALIDATION PHASE
            # val_loss = evaluate_model(model, val_dataloader, criterion, device)
            val_loss = eval_model(model=model, dataloader=test_dataloader, criterion=criterion, device=device, loss_percentage=30)
            print(f"[Feature {i}] Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            val_losses.append(val_loss)
            
            # Save the best model based on validation loss
            if val_loss < min_val_loss:
                print("New Best Model Found! Saving...")
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"features_num_directory/feature_{i}_best_validation.pth")

        # Save the final model after training
        torch.save(model.state_dict(), f"features_num_directory/feature_{i}_final.pth")

        # NOTE: Everything below is not particularly important (it's just for your convenience) since no one rly cares about train_val loss curves nor final test_loss values
        plot_train_val_loss(train_losses=train_losses, val_losses=val_losses, feature_num=i) # Plot the training and validation loss curves FOR EACH FEATURE!

        # TESTING PHASE
        # test_loss = evaluate_model(model, test_dataloader, criterion, device)
        test_loss = eval_model(model=model, dataloader=test_dataloader, criterion=criterion, device=device, loss_percentage=30)
        print(f"[Feature {i}] Final Test Loss: {test_loss:.4f}")



# NOTE: Deprecated functions
# def train_model(model, dataloader, criterion, optimizer, device):
#     """
#     Train the model for one epoch.
#     """
#     model.train()
#     epoch_loss = 0
#     for features in dataloader:
#         features = features.to(device)
#         target = features.clone()  # Clone the target to avoid accidental modification

#         # Determine how many frames to zero out
#         if features.shape[1] < 2:  # If the video consists of a single frame
#             num_zeroes = 0
#         elif features.shape[1] < 10:  # Zero out fewer frames for smaller videos
#             num_zeroes = int(np.random.uniform(0, 0.4) * features.shape[1])
#         else:
#             num_zeroes = int(np.random.uniform(0, 0.3) * features.shape[1])

#         # Randomly set some frames to 0 for the input
#         features = features.clone()  # Avoid in-place modification of the tensor
#         for _ in range(num_zeroes):
#             random_idx = np.random.randint(0, features.shape[1])
#             features[:, random_idx] = 0  # Set the frame to zeros
        
#         # Forward pass
#         output_seq = model(features)
#         loss = criterion(output_seq, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
    
#     return epoch_loss / len(dataloader)
        
# def evaluate_model(model, dataloader, criterion, device):
#     """
#     Evaluate the model on validation or test set.
#     """
#     model.eval()
#     epoch_loss = 0
#     with torch.no_grad():
#         for features in dataloader:
#             features = features.to(device)
#             target = features.clone()  # Clone the target to avoid accidental modification
#             output_seq = model(features)
#             loss = criterion(output_seq, target)
#             epoch_loss += loss.item()
#     return epoch_loss / len(dataloader)


