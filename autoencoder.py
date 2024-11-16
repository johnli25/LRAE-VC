import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (3, 224, 224) -> (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 112, 112) -> (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (128, 56, 56) -> (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (256, 28, 28) -> (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 14, 14) -> (512, 7, 7)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # (512, 7, 7) -> (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (512, 14, 14) -> (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (256, 28, 28) -> (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 56, 56) -> (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (64, 112, 112) -> (3, 224, 224)
            nn.Sigmoid()  # Ensure output is in [0, 1] range for normalized image reconstruction
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass input through the encoder
        x = self.decoder(x)  # Pass the encoded representation through the decoder
        return x
    

# AJ's quick initial autoencoder, assuming input dim is 224x224x3
class ConvAutoencoderAJ(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input: (3, 224, 224) -> Output: (64, 112, 112)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Output: (128, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Output: (256, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # Output: (512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # Output: (1024, 7, 7)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # Output: (512, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Output: (256, 28, 28)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Output: (128, 56, 56)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 112, 112)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # Output: (3, 224, 224)
            nn.Sigmoid()  # For pixel values in range [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class PNC_Autoencoder(nn.Module):
    def __init__(self):
        super(PNC_Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (16, 32, 32)
        self.encoder2 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)  # (16, 32, 32) -> (10, 32, 32)

        # Decoder
        self.decoder1 = nn.ConvTranspose2d(10, 64, kernel_size=9, stride=7, padding=4, output_padding=6)  # (10, 32, 32) -> (64, 224, 224)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (64, 224, 224) -> (3, 224, 224)

        # Activation Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For output normalization in range [0, 1]

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.encoder1(x))  # (16, 32, 32)
        # print(f"Shape after encoder1: {x1.shape}")
        x2 = self.relu(self.encoder2(x1))  # (10, 32, 32)
        # print(f"Shape after encoder2: {x2.shape}")

        # Decoder
        y1 = self.relu(self.decoder1(x2))  # (64, 224, 224)
        # print(f"Shape after decoder1: {y1.shape}")

        y2 = self.relu(self.decoder2(y1))  # (64, 224, 224)
        # print(f"Shape after decoder2: {y2.shape}")
        y2 = y2 + y1 # Skip connection

        y3 = self.relu(self.decoder3(y2))  # (64, 224, 224)
        # print(f"Shape after decoder3: {y3.shape}")

        y4 = self.relu(self.decoder3(y3))  # (64, 224, 224)
        # print(f"Shape after decoder3 (second time): {y4.shape}")
        y4 = y4 + y3  # Skip connection

        y5 = self.final_layer(y4)  # (3, 224, 224)
        # print(f"Shape after final_layer: {y5.shape}")
        y5 = torch.clamp(y5, min=0, max=1)  # Ensure output is in [0, 1] range
        return y5




# Dataset class for loading images and ground truths
class ImageDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        gt_path = os.path.join(self.gt_dir, self.img_names[idx])

        image = Image.open(img_path).convert("RGB")
        ground_truth = Image.open(gt_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        return image, ground_truth, self.img_names[idx]

#Testing and training
def train_autoencoder(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    return train_loss / len(dataloader.dataset)

def test_autoencoder(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    return test_loss / len(dataloader.dataset)


if __name__ == "__main__":
    ## Hyperparameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 1e-3
    img_height, img_width = 224, 224  # Dependent on autoencoder architecture

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"

    dataset = ImageDataset(path, path, transform=transform)

    # Define test dataset using specified filenames
    test_img_names = [
        "diving_7", "diving_8", "golf_front_7", "golf_front_8", "kick_front_8", "kick_front_9",
        "lifting_5", "lifting_6", "riding_horse_8", "riding_horse_9", "running_7", "running_8",
        "running_9", "skating_8", "skating_9", "swing_bench_7", "swing_bench_8", "swing_bench_9"
    ]

    test_indices = [
        i for i in range(len(dataset))
        if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
    ]
    train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

    # Split train_val_indices into train and validation
    np.random.shuffle(train_val_indices)
    train_size = int(0.8 * len(train_val_indices))
    val_size = len(train_val_indices) - train_size

    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_transform = transforms.Compose([
      transforms.Resize((img_height, img_width)),
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(15),
      transforms.RandomRotation([-30, 30])
      ])
    
    # apply data augmentation to the train data
    train_dataset.dataset.transform = train_transform  
    val_dataset.dataset.transform = transform        
    test_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PNC_Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with validation
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train the model
        train_loss = train_autoencoder(model, train_loader, criterion, optimizer, device)
        
        # Validate the model
        val_loss = test_autoencoder(model, val_loader, criterion, device)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_validation_autoencoder.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Final test evaluation
    test_loss = test_autoencoder(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), "autoencoder_final.pth")



    # Save images generated by decoder 
    output_path = "output_imgs/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model.eval()
    print(f"Total number of batches in test_loader: {len(test_loader)}")
    print(f"Total number of samples in test_dataset: {len(test_loader.dataset)}")
    with torch.no_grad():
        for i, (inputs, _, filenames) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            for j in range(inputs.size()[0]):
                output = outputs[j].permute(1, 2, 0).cpu().numpy() # outputs[j] original shape is (3, 224, 224), which need to convert to -> (224, 224, 3)
                # output = (output * 255).astype(np.uint8)

                # save the numpy array as image
                plt.imsave(os.path.join(output_path, filenames[j]), output)