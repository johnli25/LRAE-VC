import os
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, directory, feature_num):
        """
        Args:
            directory (str): Path to the directory containing the .npy files.
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
            tuple: (sequence_item) tensor of shape (X, 32, 32), where X is the number of sequence items in the file
        """
        file_path = self.file_paths[idx]
        
        # Load the file (assuming it's in numpy format)
        data = np.load(file_path)  # This will be of shape (10, X, 32, 32)... (128, X, 28, 28) ???
        swapped_data = np.swapaxes(data, 0, 1)
        
        # Extract the first item (you can modify this if you want to select a specific index)
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
        # Reshape input from [batch_size, sequence_length, 32, 32] -> [batch_size, sequence_length, 1024]
        batch_size, sequence_length, height, width = x.shape
        x = x.view(batch_size, sequence_length, -1)
        
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply fully connected layer to each timestep
        output = self.fc(lstm_out)
        
        # Reshape output back to [batch_size, sequence_length, 32, 32]
        output = output.view(batch_size, sequence_length, height, width)
        return output

#Testing and training
def train_feature_filling(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for features in dataloader:
        features = features.to(device)
        #print(features.size())
        target = features.clone().detach().to(device)

        # this is pretty arbitrary and can be changed
        if features.shape[1] < 2: # dont zero out any frames if the video consists of a single frame
            num_zeroes = 0
        elif features.shape[1] < 10: # zero out fewer frames for smaller videos
            num_zeroes = 1
        else:
            num_zeroes = np.random.randint(1, 6) # introduce up to 5 (?) dropped frames if we can handle it

        #randomly set some frames to 0 for our input image
        for _ in range(num_zeroes):
            random_idx = np.random.randint(0, features.shape[1])
            features[0][random_idx] = torch.zeros((features.size()[2], features.size()[3])) # THIS SHOULD BE 28x28 rn??? NEEDS TO CHANGE TO WHATEVER OUR FEATURE SSIZE IS

        output_seq = model(features)
        loss = criterion(output_seq, target)
        #print("Loss: " + str(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss
    
    return epoch_loss / len(dataloader)

def test_feature_filling(model, dataloader, criterion, optimizer, device):
    model.eval()
    epoch_loss = 0
    for features in dataloader:
        features = features.to(device)
        #print(features.size())
        target = features.clone().detach().to(device)

        # this is pretty arbitrary and can be changed
        if features.shape[1] < 2: # dont zero out any frames if the video consists of a single frame
            num_zeroes = 0
        elif features.shape[1] < 10: # zero out fewer frames for smaller videos
            num_zeroes = 1
        else:
            num_zeroes = np.random.randint(1, 6) # introduce up to 5 (?) dropped frames if we can handle it

        #randomly set some frames to 0 for our input image
        for _ in range(num_zeroes):
            random_idx = np.random.randint(0, features.shape[1])
            features[0][random_idx] = torch.zeros((features.size()[2], features.size()[3])) # THIS SHOULD BE 28x28 rn??? NEEDS TO CHANGE TO WHATEVER OUR FEATURE SSIZE IS

        output_seq = model(features)
        loss = criterion(output_seq, target)
        epoch_loss += loss
    
    return epoch_loss / len(dataloader)


epochs = 60 
folder_path = "PNC_combined_features"
batch_size = 1

# Hyperparameters
input_dim = 32 * 32  # Flattened frame size (idk if this is 32x32 or 28x28)
hidden_dim = 128 # Tuned hyperparameter for LSTM
output_dim = 32 * 32
num_layers = 2
learning_rate = 0.001
num_features = 10 # Number of latent features

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#model = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_features):
    # WILL NEED TO ITERATE HERE FOR EVERY FEATURE INDEX
    print(f"ITERATION {i}")
    model = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device) # new model for every iteration?
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = CustomDataset(folder_path, i) # the number dictates which feature we are training (right now im just training for feature 0, but we will need to do this 10 (or 128?) times)

    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)

    # 25% Testing
    test_size = len(dataset) // 4

    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    min_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_feature_filling(model, train_dataloader, criterion, optimizer, device)
        print("[EPOCH " + str(epoch) +"] Training Loss: " + str(train_loss))
        test_loss = test_feature_filling(model, test_dataloader, criterion, optimizer, device)
        print("[EPOCH " + str(epoch) +"] Testing Loss: " + str(test_loss))

        if (test_loss < min_loss):
            print("New Minimum! Saving Model")
            min_loss = test_loss
            torch.save(model.state_dict(), f"feature_{i}_best_validation.pth")
