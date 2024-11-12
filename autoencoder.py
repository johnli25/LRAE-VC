import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Quick initial autoencoder, assuming input dim is 224x224x3
class ConvAutoencoder(nn.Module):
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



# Hyperparameters
num_epochs = 40
batch_size = 32
learning_rate = 1e-3
img_height, img_width = 224, 224  # Replace with actual dimensions (dependent on autoencoder architecture)

# Data loading
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/"

dataset = ImageDataset(path, path, transform=transform)
# train_size = int(0.8 * len(dataset))  # 851
# test_size = len(dataset) - train_size  # 1064 - 851 = 213

# List of specific image names for the test dataset
test_img_names = [
    "diving_7", "diving_8", "golf_front_7", "golf_front_8", "kick_front_8", "kick_front_9",
    "lifting_5", "lifting_6", "riding_horse_8", "riding_horse_9", "running_7", "running_8",
    "running_9", "skating_8", "skating_9", "swing_bench_7", "swing_bench_8", "swing_bench_9"
]

# Create a subset of the dataset for the test dataset
test_indices = [
    i for i in range(len(dataset))
    if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
]
train_indices = [i for i in range(len(dataset)) if i not in test_indices]
test_dataset = Subset(dataset, test_indices)
train_dataset = Subset(dataset, train_indices)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_autoencoder(model, train_loader, criterion, optimizer, device)
    test_loss = test_autoencoder(model, test_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Save the model after training
torch.save(model.state_dict(), "autoencoder.pth")


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
        print(inputs.size())
        print(outputs.size())
        for j in range(inputs.size()[0]):
            output = outputs[j].permute(1, 2, 0).cpu().numpy() # outputs[j] original shape is (3, 224, 224), which need to convert to -> (224, 224, 3)
            # output = (output * 255).astype(np.uint8)

            # save the numpy array as image
            plt.imsave(os.path.join(output_path, filenames[j]), output)

