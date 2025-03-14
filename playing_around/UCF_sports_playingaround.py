import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models import ComplexCNN

# global classmap
class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

def get_labels_from_filename(filenames):
    labels = []
    for filename in filenames:
        activity = "_".join(filename.split("_")[:-2])
        labels.append(class_map[activity])
    return labels


class VideoFramesDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.filenames = os.listdir(img_dir)
        
        # group frames
        self.video_dict = {}
        for fname in self.filenames:
            splitted = fname.rsplit('.', 1)[0].split('_')
            # Example: "swing_bench_1_002"
            # splitted => ["swing", "bench", "1", "002"]

            action_str = "_".join(splitted[:-2])  # e.g. "swing_bench"
            video_id   = splitted[-2]             # e.g. "1"
            frame_str  = splitted[-1]             # e.g. "002"
            frame_num  = int(frame_str)
            video_key  = f"{action_str}_{video_id}"  # e.g. "swing_bench_1"

            if video_key not in self.video_dict:
                self.video_dict[video_key] = []
            self.video_dict[video_key].append((frame_num, fname))
        
        # sort b/c filenames/frames from os.listdir() can be arbitrarily ordered
        for k in self.video_dict:
            self.video_dict[k].sort(key=lambda x: x[0])
        
        self.video_keys = list(self.video_dict.keys())

    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        print(idx)
        video_key = self.video_keys[idx]  # e.g. "diving_7"
        frames_info = self.video_dict[video_key]  # list of (frame_num, filename)
        
        action_str = "_".join(video_key.split("_")[:-1])  # e.g. extract from "swing_bench" video key "swing_bench_7"
        print(action_str)
        label_idx = class_map[action_str]
        
        frames = []
        for frame_num, filename in frames_info:
            fpath = os.path.join(self.img_dir, filename)
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        frames_tensor = torch.stack(frames, dim=0)  # shape (T, C, H, W)
        return frames_tensor, label_idx


class VideoTransformer(nn.Module):
    def __init__(self, num_classes=101, embed_dim=512, num_heads=8, num_layers=4):
        super(VideoTransformer, self).__init__()
        self.embedding = nn.Linear(128 * 128 * 3, embed_dim)  # Flattened frame embedding
        self.positional_encoding = nn.Parameter(torch.randn(1, 64, embed_dim))  # Positional encoding for 64 frames
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, frames, channels, height, width)
        x = x.float()
        print("x size", x.size())
        batch_size, frames, c, h, w = x.size()
        x = x.reshape(batch_size, frames, -1)  # Flatten each frame
        x = self.embedding(x) + self.positional_encoding[:, :frames, :]
        x = x.transpose(0, 1)  # (frames, batch, embed_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, frames, embed_dim)
        x = x.mean(dim=1)      # average over frames
        return self.fc(x)

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, scheduler=None):
    def plot_train_val_loss(train_losses, val_losses):
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_losses, label='Training Loss', color='blue', marker='o')
        plt.plot(epochs_range, val_losses, label='Validation Loss', color='orange', marker='o')
        plt.title('Training and Validation Loss', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('train_val_loss_curve.png', dpi=300)
        plt.show()

    model.train()
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, filenames in train_loader:
            print("filenames", filenames)
            labels = torch.tensor(get_labels_from_filename(filenames))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        if scheduler:
            scheduler.step()
        train_loss_epoch = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss_epoch)
        val_loss_epoch = test(model, val_loader, criterion, device)
        val_losses.append(val_loss_epoch)
        if val_loss_epoch < min(val_losses, default=float('inf')):
            torch.save(model.state_dict(), 'best_classifier_model.pth')
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss_epoch:.4f} - Model saved!")
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}")
    plot_train_val_loss(train_losses, val_losses)

def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, filenames in dataloader:
            labels = torch.tensor(get_labels_from_filename(filenames))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    return test_loss / len(dataloader)

def calculate_accuracy(loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def custom_collate(batch): # NOTE: currently not used b/c no audio in my custom UCF Sports Action Dataset!!!
    """
    Filter out the audio, returning only (video, label).
    """
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


if __name__ == "__main__":
    ## Hyperparameters
    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-3
    img_height, img_width = 224, 224
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"  # Update this path appropriately

    # Data loading transforms
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset instance (only one argument for img_dir)
    dataset = VideoFramesDataset(img_dir=path, transform=transform)
    # Define test dataset using specified filenames
    test_video_names = {
        "diving_7", "golf_front_8", "kick_front_8", 
        "lifting_5", "riding_horse_9", "running_7",
        "running_9", "skating_8", "swing_bench_7",
    }
    
    # Build test indices from the dataset's video_keys
    test_indices = [i for i, video_key in enumerate(dataset.video_keys) if video_key in test_video_names]
    # Use the remaining indices for train+validation
    train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

    print("Train/Validation indices count:", len(train_val_indices))
    print("Test indices:", test_indices)
    
    # Further split train_val_indices into training and validation sets (e.g., 80-20 split)
    np.random.shuffle(train_val_indices)
    train_size_split = int(0.9 * len(train_val_indices))
    val_size_split = len(train_val_indices) - train_size_split

    train_indices = train_val_indices[:train_size_split]
    val_indices = train_val_indices[train_size_split:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model (using VideoTransformer here; swap with ComplexCNN for CNN baseline)
    model = VideoTransformer(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, scheduler)

    # Test the model
    test_loss = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    test_acc = calculate_accuracy(test_loader, model, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
