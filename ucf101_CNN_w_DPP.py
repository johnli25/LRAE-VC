import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Resize
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoTransformer, VideoMAEFeatureExtractor 

# TODO: run on BOTH vms' GPUs: 1) this .py for 100 imgs (saved as ACTUAL images so create a new folder + comment out video processing function) 2) this .py for a few videos for the SAME 100 imgs (use video processing function)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class VideoTransformer(nn.Module):
    """
    Video Transformer that operates on flattened frames of size 64x64x3.
    """
    def __init__(self, num_classes=101, embed_dim=512, num_heads=8, num_layers=4):
        super(VideoTransformer, self).__init__()
        self.embedding = nn.Linear(64 * 64 * 3, embed_dim)  # 12288 → embed_dim
        self.positional_encoding = nn.Parameter(torch.randn(1, 64, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, frames, channels, height, width)
        x = x.float()
        batch_size, frames, c, h, w = x.size()
        x = x.reshape(batch_size, frames, -1)
        x = self.embedding(x) + self.positional_encoding[:, :frames, :]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x.mean(dim=1)
        return self.fc(x)

class VideoCNNClassifier(nn.Module):
    """
    A simple 2D CNN that averages frames over time and then classifies.
    Input frames are 64x64; two conv-pool blocks.
    """
    def __init__(self, num_classes=101):
        super(VideoCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, frames, channels, height, width)
        x = x.float()
        x = x.mean(dim=1)  # Average over frames → (batch_size, 3, 64, 64)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_collate(batch):
    """Filter out the audio, returning only (video, label)."""
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def train_cnn(rank, world_size, model, train_loader, test_loader, epochs=10, lr=0.001, save_dir='./checkpoints', resume_epoch=0):
    setup(rank, world_size)

    # NOTE: If resuming from a checkpoint, load state dict (only rank 0 needs to load, then send to other GPUs/ranks).
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{resume_epoch}.pth')
    if resume_epoch > 0 and os.path.exists(checkpoint_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint)
        print(f"[GPU {rank}] Loaded checkpoint from {checkpoint_path}")
    else:
        if rank == 0 and resume_epoch > 0:
            print(f"Checkpoint {checkpoint_path} not found; starting from scratch.")

    # NOTE: Update the sampler's rank for the current process.
    train_loader.sampler.rank = rank
    test_loader.sampler.rank = rank

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = resume_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        # set up tqdm for rank=0 progress bar
        if rank == 0: loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        else: loop = train_loader

        for videos, labels in loop:
            videos = videos.to(rank)
            labels = labels.to(rank)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[GPU {rank}] Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # NOTE: IMPORTANT-save checkpoint only from rank 0 as rank/device 0 is the master 
        if rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    cleanup()

def test_cnn(rank, world_size, model, test_loader, load_model_path=None):
    setup(rank, world_size)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    if load_model_path is not None and os.path.exists(load_model_path):
        model.load_state_dict(torch.load(load_model_path, map_location=f"cuda:{rank}"))
        print(f"[GPU {rank}] Loaded model from {load_model_path}")
    
    criterion = nn.CrossEntropyLoss().to(rank)    
    # No need for an optimizer if we are just testing
    model.eval()
    
    test_loader.sampler.rank = rank
    test_loader.sampler.set_epoch(0)

    running_loss = 0.0
    correct = 0
    total = 0

    # Conditionally wrap with tqdm on rank 0:
    if rank == 0: loop = tqdm(test_loader, desc="Testing", leave=False)
    else: loop = test_loader

    with torch.no_grad():
        for videos, labels in loop:
            videos = videos.to(rank)
            labels = labels.to(rank)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Aggregate partial results across ranks
    loss_tensor    = torch.tensor([running_loss], device=rank, dtype=torch.float)
    correct_tensor = torch.tensor([correct],      device=rank, dtype=torch.float)
    total_tensor   = torch.tensor([total],        device=rank, dtype=torch.float)

    dist.all_reduce(loss_tensor,    op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor,   op=dist.ReduceOp.SUM)

    global_loss    = loss_tensor.item()
    global_correct = correct_tensor.item()
    global_total   = total_tensor.item()

    avg_loss  = global_loss / global_total
    accuracy  = global_correct / global_total

    if rank == 0:
        print(f"[GPU {rank}] Global Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    cleanup()


def main():
    world_size = 2  # NOTE: john's VM have 2 GPUs. Adjust if on different hardware.
    batch_size = 32
    epochs = 10
    resume_epoch = 10
    lr = 0.001

    transform = Compose([Resize((64, 64))])
    full_dataset = UCF101(
        root='./UCF101/UCF-101/',
        annotation_path='./UCF101/ucfTrainTestlist',
        frames_per_clip=16,
        step_between_clips=2,
        train=True,
        transform=transform,
        output_format="TCHW"
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create DistributedSamplers with a dummy rank (0). They will be updated in each spawned process.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=0, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=0, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True
    )

    model = VideoCNNClassifier(num_classes=101)

    mp.spawn( # Train
        train_cnn,
        args=(world_size, model, train_loader, test_loader, epochs, lr, './checkpoints', resume_epoch),
        nprocs=world_size,
        join=True
    )

    mp.spawn( # Test
        test_cnn,
        args=(world_size, model, test_loader, './checkpoints/model_epoch_10.pth'),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    main()
