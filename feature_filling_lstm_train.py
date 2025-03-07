import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from models import FrameSequenceLSTM
import argparse

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


def eval_model_w_totally_random_loss(model, dataloader, criterion, device):
    """
    Evaluate the model on validation or test set.
    """
    model.eval()
    epoch_loss = 0
    total_elements = 0
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            target = features.clone()
            output_seq = model(features)
            loss = criterion(output_seq, target)  # Shape: [batch_size, ...]
            
            # Aggregate the loss manually
            batch_loss = loss.sum()  # Sum all elements in the loss tensor
            epoch_loss += batch_loss.item()
            
            # Keep track of the total number of elements
            total_elements += loss.numel()
    return epoch_loss / total_elements  # Mean loss


def parse_args():
    parser = argparse.ArgumentParser(description="Get features of desired model")
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_16", "LRAE_VC", "TestNew", "TestNew2", "TestNew3"], 
                        help="Model to train")
    return parser.parse_args()


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

    for i in range(num_features):
        # TRAINING FOR FEATURE INDEX i
        print(f"Training for Feature {i}")
        model = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
        # criterion = nn.MSELoss() # NOTE: to use train_model()
        criterion = nn.MSELoss(reduction='none') # NOTE: to use train_mode_with_mask()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        dataset = CustomDataset(folder_path, i)
        
        # Shuffle dataset indices and create splits: 70% train, 15% validation, 15% test
        all_indices = list(range(len(dataset)))
        np.random.shuffle(all_indices)
        train_end = int(0.7 * len(dataset))
        val_end = int(0.85 * len(dataset))
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

        for epoch in range(epochs):
            # TRAINING PHASE
            # train_loss = train_model(model, train_dataloader, criterion, optimizer, device) # NOTE: to use train_model()
            train_loss = train_model_w_totally_random_loss(model, train_dataloader, criterion, optimizer, device) # NOTE: to use train_model_with_mask()
            print(f"[Feature {i}] Epoch {epoch}, Training Loss: {train_loss:.4f}")
            
            # VALIDATION PHASE
            # val_loss = evaluate_model(model, val_dataloader, criterion, device)
            val_loss = eval(model, val_dataloader, criterion, device)
            print(f"[Feature {i}] Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            
            # Save the best model based on validation loss
            if val_loss < min_val_loss:
                print("New Best Model Found! Saving...")
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"features_num_directory/feature_{i}_best_validation.pth")

        # TESTING PHASE
        # test_loss = evaluate_model(model, test_dataloader, criterion, device)
        test_loss = eval_model_w_totally_random_loss(model, val_dataloader, criterion, device)
        print(f"[Feature {i}] Final Test Loss: {test_loss:.4f}")

        # Save the final model after training
        torch.save(model.state_dict(), f"features_num_directory/feature_{i}_final.pth")




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


