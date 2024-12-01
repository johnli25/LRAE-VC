import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim

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

class FrameSequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FrameSequenceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to project hidden state to output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, 32, 32].

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Reshape input from [batch_size, sequence_length, 32, 32] -> [batch_size, sequence_length, 1024]
        batch_size, sequence_length, height, width = x.shape
        x = x.view(batch_size, sequence_length, -1)  # Flatten spatial dimensions
        
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        print("h0 and c0 shape: ", h0.shape, c0.shape)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        print("lstm_out shape: ", lstm_out.shape)
        
        # Apply fully connected layer to each timestep
        output = self.fc(lstm_out)
        
        # Reshape output back to [batch_size, sequence_length, 32, 32]
        output = output.view(batch_size, sequence_length, height, width)
        print("output shape: ", output.shape)
        return output

def train_model(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_loss = 0
    for features in dataloader:
        features = features.to(device)
        target = features.clone()  # Clone the target to avoid accidental modification

        # Determine how many frames to zero out
        if features.shape[1] < 2:  # If the video consists of a single frame
            num_zeroes = 0
        elif features.shape[1] < 10:  # Zero out fewer frames for smaller videos
            num_zeroes = 1
        else:
            num_zeroes = np.random.randint(1, 6)  # Randomly drop up to 5 frames

        # Randomly set some frames to 0 for the input
        features = features.clone()  # Avoid in-place modification of the tensor
        for _ in range(num_zeroes):
            random_idx = np.random.randint(0, features.shape[1])
            features[:, random_idx] = 0  # Set the frame to zeros
        
        # Forward pass
        print("in train_model(): ", features.shape)
        output_seq = model(features)
        loss = criterion(output_seq, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on validation or test set.
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            target = features.clone()  # Clone the target to avoid accidental modification
            output_seq = model(features)
            loss = criterion(output_seq, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Configuration
epochs = 60
folder_path = "PNC_combined_features"
batch_size = 1
input_dim = 32 * 32  # Flattened frame size
hidden_dim = 128  # Tuned hyperparameter for LSTM
output_dim = 32 * 32  # Same as input_dim for reconstruction
num_layers = 2
learning_rate = 0.001
num_features = 10  # Number of latent features to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for i in range(num_features):
    # TRAINING FOR FEATURE INDEX i
    print(f"Training for Feature {i}")
    model = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.MSELoss()
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
        train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        print(f"[Feature {i}] Epoch {epoch}, Training Loss: {train_loss:.4f}")
        
        # VALIDATION PHASE
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"[Feature {i}] Epoch {epoch}, Validation Loss: {val_loss:.4f}")
        
        # Save the best model based on validation loss
        if val_loss < min_val_loss:
            print("New Best Model Found! Saving...")
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"feature_{i}_best_validation.pth")

    # TESTING PHASE
    test_loss = evaluate_model(model, test_dataloader, criterion, device)
    print(f"[Feature {i}] Final Test Loss: {test_loss:.4f}")

    # Save the final model after training
    torch.save(model.state_dict(), f"feature_{i}_final.pth")
