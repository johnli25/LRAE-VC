import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Resize, ToTensor
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEFeatureExtractor
import matplotlib.pyplot as plt

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def custom_collate(batch):
    # Each item is (video, _, label); we drop the audio and return (video, label)
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def train_videomae(rank, world_size, model, feature_extractor, train_loader, test_loader, epochs=10, lr=5e-5, save_dir='./checkpoints', resume_epoch=0):
    setup(rank, world_size)

    # Load checkpoint if resuming (only rank 0 needs to load, then send to other GPUs)
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{resume_epoch}.pth')
    if resume_epoch > 0 and os.path.exists(checkpoint_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint)
        print(f"[GPU {rank}] Loaded checkpoint from {checkpoint_path}")
    else:
        if rank == 0 and resume_epoch > 0:
            print(f"Checkpoint {checkpoint_path} not found; starting from scratch.")

    # Update sampler rank for the current process
    train_loader.sampler.rank = rank
    # test_loader.sampler.rank = rank 

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = resume_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        if rank == 0:
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        else:
            loop = train_loader

        for videos, labels in loop:
            # UCF101 with output_format="THWC" with Batch size B added to dim = 0
            # thus, we need to permute from (B, T, H, W, C) to (B, C, T, H, W) 
            videos = videos.permute(0, 4, 1, 2, 3)
            videos_list = [] # Convert each video to a list of frames (each frame: (C, H, W))
            for video in videos:
                # video: (C, T, H, W) -> list of T frames each of shape (C, H, W)
                frames = [video[:, i, :, :] for i in range(video.shape[1])]
                videos_list.append(frames)
            inputs = feature_extractor(
                images=videos_list, 
                return_tensors="pt", 
                input_data_format="channels_first"
            )["pixel_values"].to(rank)

            labels = labels.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs).logits  # VideoMAE returns an output object; we use .logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * videos.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[GPU {rank}] Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        if rank == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, f'VideoMAE_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    cleanup()


def test_videomae(rank, world_size, model, feature_extractor, test_loader):
    setup(rank, world_size)

    test_loader.sampler.rank = rank
    test_loader.sampler.set_epoch(0)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()
    criterion = nn.CrossEntropyLoss().to(rank)

    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in test_loader:
            # UCF101 with output_format="THWC" with Batch size B added to dim = 0
            # thus, we need to permute from (B, T, H, W, C) to (B, C, T, H, W) 
            videos = videos.permute(0, 4, 1, 2, 3)
            videos_list = [] # Convert each video to a list of frames (each frame: (C, H, W))
            for video in videos:
                # video: (C, T, H, W) -> list of T frames each of shape (C, H, W)
                frames = [video[:, i, :, :] for i in range(video.shape[1])]
                videos_list.append(frames)
            inputs = feature_extractor(
                images=videos_list, 
                return_tensors="pt", 
                input_data_format="channels_first"
            )["pixel_values"].to(rank)
            
            labels = labels.to(rank)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            test_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f"[GPU {rank}] Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    cleanup()


def main():
    world_size = 2  # NOTE: john's VM have 2 GPUs. Adjust if on different hardware.
    batch_size = 8  # Batch size per GPU (adjust for memory)
    epochs = 5
    resume_epoch = 0
    lr = 3e-5

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base") # Use the VideoMAE feature extractor (preprocessing pipeline)

    transform = Compose([]) # NOTE: VideoMAEFeatureExtractor will automatically resize the frames to 224x224 (so remove `Resize((224, 224))` for now)
    full_dataset = UCF101(
        root='./UCF101/UCF-101/',
        annotation_path='./UCF101/ucfTrainTestlist',
        frames_per_clip=16,
        step_between_clips=2,
        train=True,
        transform=transform,
        output_format="THWC" # NOTE: UNFORTUNATELY, UCF101 dataset only supports "THWC" format (instead of CTHW which is what VideoMAE feature extractor/model expects)
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

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

    # Load a pretrained VideoMAE model for vdideo classification (fine-tuned on Kinetics)
    model = VideoMAEForVideoClassification.from_pretrained(
        "nateraw/videomae-base-finetuned-ucf101", num_labels=101
        # "MCG-NJU/videomae-base-finetuned-kinetics", num_labels=101 
    )

    mp.spawn(
        train_videomae,
        args=(world_size, model, feature_extractor, train_loader, test_loader, epochs, lr, './checkpoints', resume_epoch),
        nprocs=world_size,
        join=True
    )

    mp.spawn(
        test_videomae,
        args=(world_size, model, feature_extractor, test_loader),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    main()
