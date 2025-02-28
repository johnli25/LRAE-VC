import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Resize
import subprocess

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

class VideoTransformer(nn.Module):
    """
    Video Transformer that operates on flattened frames of size 64x64x3.
    """
    def __init__(self, num_classes=101, embed_dim=512, num_heads=8, num_layers=4):
        super(VideoTransformer, self).__init__()
        # Flattened frame size = 64 * 64 * 3 = 12288
        self.embedding = nn.Linear(64 * 64 * 3, embed_dim)
        # Let's allow up to 64 frames in a clip for positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 64, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: shape (batch_size, frames, channels, height, width)
        """
        x = x.float()
        batch_size, frames, c, h, w = x.size()
        
        # Flatten each frame => (batch_size, frames, 64*64*3)
        x = x.reshape(batch_size, frames, -1)
        
        # Embedding + positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :frames, :]

        # Transformer expects (sequence_length, batch_size, embed_dim)
        x = x.transpose(0, 1)  # => (frames, batch_size, embed_dim)
        
        # Pass through the Transformer
        x = self.transformer(x)  # => (frames, batch_size, embed_dim)
        
        x = x.transpose(0, 1)    # => (batch_size, frames, embed_dim)
        x = x.mean(dim=1)        # average over frames

        return self.fc(x)

class VideoCNNClassifier(nn.Module):
    """
    A simple 2D CNN that averages frames over time and then classifies.
    Input frames are 64x64, then we do two conv-pool blocks.
    """
    def __init__(self, num_classes=101):
        super(VideoCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After two pooling layers, 64x64 becomes 16x16. With 64 channels, that gives 64 * 16 * 16 = 16384 features.
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, frames, channels, height, width)
        x = x.float()
        # Average over time dimension (frames)
        x = x.mean(dim=1)  # Now: (batch_size, 3, 64, 64)
        x = torch.relu(self.conv1(x))   # (batch_size, 32, 64, 64)
        x = self.pool(x)                # (batch_size, 32, 32, 32)
        x = torch.relu(self.conv2(x))   # (batch_size, 64, 32, 32)
        x = self.pool(x)                # (batch_size, 64, 16, 16)
        x = x.view(x.size(0), -1)       # Flatten: (batch_size, 16384)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_collate(batch):
    """Filter out the audio, returning only (video, label)."""
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def train_cnn(rank, world_size, epochs=10, lr=0.001):
    # Typically you want one process per GPU. Here we assume that each process sees one local GPU.
    # NOTE: setup + print checks for multiple device(s)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "N/A"

    print(f"Device type: {device_type}")
    print(f"Device count: {device_count}")
    print(f"Current device: {current_device}")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        print(f"Process {rank} using device {torch.cuda.current_device()}")
        device = torch.device("cuda", rank % device_count)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank) # The 'env://' method expects SLURM (or your launcher) to set WORLD_SIZE and RANK.

    # Set up transforms and dataset.
    transform = Compose([Resize((64, 64))])
    full_dataset = UCF101(
        root='./UCF101_DATA/UCF-101/',
        annotation_path='./UCF101_DATA/ucfTrainTestlist',
        frames_per_clip=16,    # Adjust if desired for faster training
        step_between_clips=2,  # Can be increased to reduce the number of clips
        train=True,
        transform=transform,
        output_format="TCHW"
    )

    # Split into train/test splits (this split is performed on every process,
    # but the DistributedSampler will ensure that each process sees a unique subset).
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Use DistributedSampler to partition the data among all processes.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler  = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=train_sampler,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        sampler=test_sampler,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True
    )

    # Create and wrap the model with DistributedDataParallel.
    model = VideoCNNClassifier(num_classes=101).to(device)
    model = DDP(model, device_ids=[rank % torch.cuda.device_count()])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("train and test loaders", len(train_loader), len(test_loader))
    for epoch in range(epochs):
        model.train()
        # Set the epoch for the sampler for proper shuffling.
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for videos, labels in train_loader:
            print(len(videos), labels)
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)
        epoch_loss = running_loss / len(train_sampler)

        # Print from the rank 0 process to avoid duplicate logs:
        if rank == 0: print(f"[CNN] Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}") 

    # Evaluate the model (each process computes its own portion).
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # NOTE: Optionally, aggregate accuracy across processes. Here we simply print from rank 0.
    if rank == 0:
        accuracy = 100 * correct / total
        print(f"[CNN] Final Test Accuracy: {accuracy:.2f}%")

    dist.destroy_process_group()

if __name__ == '__main__':
    # For example, if you request --nodes=4 and --ntasks-per-node=4, then the total world size is 16.
    world_size = int(os.environ.get("WORLD_SIZE", 16))  # total number of processes
    rank = int(os.environ.get("RANK", 0))  # unique rank of this process
    train_cnn(rank, world_size, epochs=10, lr=0.001)

