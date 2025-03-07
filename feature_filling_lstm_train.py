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

        # Extract the specified feature sequence
        sequence_item = swapped_data[self.num]  
        
        # Convert to a torch tensor
        return torch.tensor(sequence_item, dtype=torch.float32)


def train_model_w_totally_random_loss(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch_idx, features in enumerate(dataloader):
        features = features.to(device)
        target = features.clone()  # Clone the target to avoid accidental modification

        # Determine which frames to zero out based on drop_probability
        sequence_length = features.shape[1]
        if sequence_length < 2: # very short video lol
            drop_probability = 0.0  # Don't drop/mask anything
        elif sequence_length < 10:
            drop_probability = 0.15  # Drop up to 15%
        else:
            drop_probability = 0.25  # Drop 25%
        mask = torch.rand(sequence_length, device=device) < drop_probability  # Shape: [sequence_length]
        
        if mask.sum() == 0:
            # If no frames are dropped, skip this batch
            continue
        
        # Expand mask to match the feature dimensions
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(features)  # Shape: [batch_size, sequence_length, 32, 32] (Depends on model)
        
        # Zero out the selected frames in the input
        input_features = features.clone()
        input_features[mask_expanded] = 0  # Set the zeroed frames to zero

        # Forward pass
        output_seq = model(input_features)

        # Compute loss only on zeroed-out frames
        # Use 'reduction="none"' to compute element-wise loss
        loss = criterion(output_seq, target)  # Shape: [batch_size, sequence_length, 32, 32]
        
        # Apply the mask to the loss
        masked_loss = loss * mask_expanded.float()  # Only keep loss for zeroed frames
        
        # Compute the mean loss over the number of zeroed frames
        print("Total sum of masked loss and # of masked/dropped feature tensors ", masked_loss.sum(), mask_expanded.float().sum())
        loss_value = masked_loss.sum() / mask_expanded.float().sum()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss_value.item()
    
    # Compute average loss over all batches
    average_loss = epoch_loss / len(dataloader)
    return average_loss



def train_model_grace_loss(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    loss_rates = [10, 20, 30, 40, 50, 60]

    for batch_idx, features in enumerate(dataloader):
        features = features.to(device)
        target = features.clone()  # Clone the target to avoid accidental modification

        sequence_length = features.shape[1]

        # -- GRACE-inspired packet loss sampling --
        if np.random.rand() < 0.8: loss_percentage = 0  # 80% chance it's a clean frame
        else: loss_percentage = np.random.choice(loss_rates)  # 20% chance, choose from predefined losses

        # -- Create the mask based on the selected loss percentage --
        num_zeroed_frames = int((loss_percentage / 100.0) * sequence_length)

        # Create mask (zero out exactly `num_zeroed_frames` random frames)
        mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        if num_zeroed_frames > 0:
            zero_indices = np.random.choice(sequence_length, num_zeroed_frames, replace=False)
            mask[zero_indices] = True

        # Expand mask to match full feature shape (batch_size, sequence_length, height, width)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(features)

        # Zero out the selected frames in the input
        input_features = features.clone()
        input_features[mask_expanded] = 0  # Apply masking (simulated packet loss)

        # Forward pass
        output_seq = model(input_features)

        # Compute per-frame loss (reduction="none" should give shape [batch_size, sequence_length, height, width])
        loss = criterion(output_seq, target)

        # Apply mask to the loss so only dropped frames contribute
        masked_loss = loss * mask_expanded.float()

        if mask_expanded.float().sum() == 0: # Edge case: If the frame was clean (0% loss), skip loss calculation for this batch
            continue

        # Average loss only over dropped frames (following Grace's focus on corrupted data recovery)
        loss_value = masked_loss.sum() / mask_expanded.float().sum()

        # Backprop and optimizer step
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # Accumulate for epoch average loss
        epoch_loss += loss_value.item()

    # Average loss over batches
    average_loss = epoch_loss / len(dataloader)
    return average_loss


def eval_model_masked_loss_only(model, dataloader, criterion, device, loss_percentage): # NOTE: fixed/consistent loss
    """Evaluate the model under a FIXED packet loss rate (GRACE-style evaluation) and compute avg loss only on dropped frames!"""
    model.eval()
    epoch_loss = 0
    total_elements = 0

    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            target = features.clone()

            sequence_length = features.shape[1]
            num_zeroed_frames = int((loss_percentage / 100.0) * sequence_length)

            mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
            if num_zeroed_frames > 0:
                zero_indices = np.random.choice(sequence_length, num_zeroed_frames, replace=False)
                mask[zero_indices] = True

            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(features)
            input_features = features.clone()
            input_features[mask_expanded] = 0 # Apply packet loss mask

            output_seq = model(input_features)
            loss = criterion(output_seq, target)

            masked_loss = loss * mask_expanded.float()  # Only compute loss on dropped frames (optional)

            batch_loss = masked_loss.sum() # Sum loss for dropped frames
            epoch_loss += batch_loss.item()
            total_elements += mask_expanded.float().sum().item()

    return epoch_loss / total_elements

def parse_args():
    parser = argparse.ArgumentParser(description="Get features of desired model")
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_16", "LRAE_VC", "TestNew", "TestNew2", "TestNew3"], 
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
    epochs = 160
    if args.model == "PNC":
        input_dim = 32 * 32
        output_dim = 32 * 32
        num_features = 10    
        folder_path = "PNC_combined_features"
    if args.model == "PNC_16":
        input_dim = 32 * 32
        output_dim = 32 * 32
        num_features = 16
        folder_path = "PNC_16_combined_features"
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
            # train_loss = train_model(model, train_dataloader, criterion, optimizer, device) # NOTE: to use train_model()
            train_loss = train_model_grace_loss(model, train_dataloader, criterion, optimizer, device) # NOTE: to use train_model_with_mask()
            print(f"[Feature {i}] Epoch {epoch}, Training Loss: {train_loss:.4f}")
            train_losses.append(train_loss)
            
            # VALIDATION PHASE
            # val_loss = evaluate_model(model, val_dataloader, criterion, device)
            val_loss = eval_model_masked_loss_only(model=model, dataloader=test_dataloader, criterion=criterion, device=device, loss_percentage=30)
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
        test_loss = eval_model_masked_loss_only(model=model, dataloader=test_dataloader, criterion=criterion, device=device, loss_percentage=30)
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


