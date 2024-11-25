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
import argparse
from transformers import FlavaProcessor, FlavaModel



class PNC_Autoencoder(nn.Module):
    def __init__(self):
        super(PNC_Autoencoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)
        self.encoder2 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder1 = nn.ConvTranspose2d(10, 64, kernel_size=9, stride=7, padding=4, output_padding=6)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1,  padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, tail_length=None):
        # Encoder
        x1 = self.relu(self.encoder1(x))
        x2 = self.relu(self.encoder2(x1))

        if tail_length is not None:
            # Zero out tail features for all samples in the batch
            batch_size, channels, _, _ = x2.size()
            tail_start = channels - tail_length
            x2 = x2.clone() # Create a copy of the tensor to avoid in-place operations!
            x2[:, tail_start:, :, :] = 0

        # Decoder
        y1 = self.relu(self.decoder1(x2))
        y2 = self.relu(self.decoder2(y1))
        y2 = y2 + y1  # Skip connection
        y3 = self.relu(self.decoder3(y2))
        y4 = self.relu(self.decoder3(y3))
        y4 = y4 + y3  # Skip connection
        y5 = self.final_layer(y4)
        y5 = torch.clamp(y5, min=0, max=1)

        return y5
    


class LRAE_VC_Autoencoder(nn.Module):
    def __init__(self):
        super(LRAE_VC_Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # (3, 224, 224) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (32, 112, 112) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 56, 56) -> (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 28, 28) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (64, 56, 56) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (32, 112, 112) -> (16, 224, 224)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        self.final_layer = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # (16, 224, 224) -> (3, 224, 224)

        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)  # (32, 112, 112)
        x2 = self.encoder2(x1)  # (64, 56, 56)
        x3 = self.encoder3(x2)  # (128, 28, 28)

        # Decoder
        y1 = self.decoder1(x3)  # (64, 56, 56)
        y1 = y1 + x2  # Skip connection

        y2 = self.decoder2(y1)  # (32, 112, 112)
        y2 = y2 + x1  # Skip connection

        y3 = self.decoder3(y2)  # (16, 224, 224)

        y4 = self.final_layer(self.dropout(y3))  # (3, 224, 224)
        y5 = self.sigmoid(y4)  # Normalize output to [0, 1]
        return y5



class PNC_Autoencoder_with_Classification(nn.Module):
    def __init__(self, num_classes=10, classes=['diving', 'golf_front', 'kick_front', 'lifting', 'riding_horse', 'running', 'skating', 'swing_bench', 'swing_side', 'walk_front']): # Default classes derived from UCF-101
        super(PNC_Autoencoder_with_Classification, self).__init__()

        self.classes = classes

        # Encoder
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (16, 32, 32)
        self.encoder2 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)  # (16, 32, 32) -> (10, 32, 32)

        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 32 * 32, 256),  # Flattened bottleneck size -> 256 hidden units
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Final layer for classification
        )

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
        x2 = self.relu(self.encoder2(x1))  # (10, 32, 32)

        # Decoder
        y1 = self.relu(self.decoder1(x2))  # (64, 224, 224)
        y2 = self.relu(self.decoder2(y1))  # (64, 224, 224)
        y2 = y2 + y1 # Skip connection
        y3 = self.relu(self.decoder3(y2))  # (64, 224, 224)
        y4 = self.relu(self.decoder3(y3))  # (64, 224, 224)
        y4 = y4 + y3  # Skip connection
        y5 = self.final_layer(y4)  # (3, 224, 224)
        y5 = torch.clamp(y5, min=0, max=1)  # Ensure output is in [0, 1] range

        # Classification
        class_scores = self.classifier(x2)  # (10, 32, 32) -> (num_classes)

        return y5, class_scores # Return (decoded image, class output label)



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


# Testing and training
def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, max_tail_length):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            # Sample a single tail length for the batch
            torch.manual_seed(seed=42)
            tail_len = torch.randint(0, max_tail_length + 1, (1,)).item()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, tail_length=tail_len) # NOTE: set tail_length=None to use the full sequence OR set tail_length=tail_length to enable stochastic tail-dropout
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        val_loss = test_autoencoder(model, val_loader, criterion, device, max_tail_length)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_name}_best_validation.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Final Test
    test_loss = test_autoencoder(model, test_loader, criterion, device, max_tail_length)
    print(f"Final Test Loss: {test_loss:.4f}")

    plot_train_val_loss(train_losses, val_losses)

    # Save final model
    torch.save(model.state_dict(), f"{model_name}_final.pth")


def test_autoencoder(model, dataloader, criterion, device, max_tail_length):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            torch.manual_seed(seed=41)
            tail_len = torch.randint(0, max_tail_length + 1, (1,)).item()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, tail_length=tail_len) # NOTE: set tail_length=None to use the full sequence OR set tail_length=tail_length to enable stochastic tail-dropout
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    return test_loss / len(dataloader.dataset)





def train_combined_model(model, train_loader, val_loader, test_loader, criterion_reconstruction, criterion_classification, optimizer, device, num_epochs, model_name):
    print(f"Training {model_name} with combined loss...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0
        cnt = 1
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, class_scores = model(inputs)

            # Reconstruction loss
            loss_reconstruction = criterion_reconstruction(outputs, targets)

            # Classification loss
            labels = get_labels_from_filename(_)
            labels = torch.tensor(labels).to(device)
            loss_classification = criterion_classification(class_scores, labels)

            # Combined loss
            loss = loss_reconstruction + loss_classification
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            print(cnt)
            cnt += 1

        train_loss /= len(train_loader.dataset)

        # Validate the model
        val_loss = test_combined_model(model, val_loader, criterion_reconstruction, criterion_classification, device)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_name}_best_validation.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        torch.cuda.empty_cache()

    # Final Testing
    test_loss = test_combined_model(model, test_loader, criterion_reconstruction, criterion_classification, device)
    print(f"Final Test Loss for {model_name}: {test_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), f"{model_name}_final.pth")

def test_combined_model(model, dataloader, criterion_reconstruction, criterion_classification, device, final_test=False):
    model.eval()
    test_loss = 0
    all_predictions = []
    filenames_list = []
    with torch.no_grad():
        for inputs, targets, img_file_name in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, class_scores = model(inputs)

            # Reconstruction loss
            loss_reconstruction = criterion_reconstruction(outputs, targets)

            # Classification loss
            labels = get_labels_from_filename(img_file_name)
            labels = torch.tensor(labels).to(device)
            loss_classification = criterion_classification(class_scores, labels)

            # Combined loss
            loss = loss_reconstruction + loss_classification
            test_loss += loss.item() * inputs.size(0)

            # Predict class labels
            _, predicted_labels = torch.max(class_scores, 1) # shape is (batch_size,) since it's batch_size of predictions
            all_predictions.extend(predicted_labels.cpu().numpy()) # ... that's why use .extend() because it's a list of predictions
            filenames_list.extend(img_file_name)

    if final_test:
        # Print predictions
        for filename, prediction in zip(filenames_list, all_predictions):
            print(f"Image: {filename}, Predicted Label: {prediction}")

    return test_loss / len(dataloader.dataset)



def get_labels_from_filename(filenames, class_map = {
        "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
        "running": 5, "skating": 6, "swing_bench": 7, "swing_side": 8, "walk_front": 9
    }):
    labels = []
    for filename in filenames:
        activity = "_".join(filename.split("_")[:-2])
        labels.append(class_map[activity])
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_with_classification", "LRAE_VC"], 
                        help="Model to train")
    return parser.parse_args()


def plot_train_val_loss(train_losses, val_losses):
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

    plt.savefig('train_val_loss_curve.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    args = parse_args()

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
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform        
    test_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    if args.model == "PNC":
        model = PNC_Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        max_tail_length = 10
        train_autoencoder(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, args.model, max_tail_length)
        
    if args.model == "LRAE_VC":
        model = LRAE_VC_Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_autoencoder(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, args.model)
    
        # Call the respective training function for autoencoder
    if args.model == "PNC_with_classification":
        model = PNC_Autoencoder_with_Classification().to(device)
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_combined_model(model, train_loader, val_loader, test_loader, criterion_reconstruction, criterion_classification, optimizer, device, num_epochs, args.model)



    # Save images generated by decoder 
    output_path = "output_test_imgs_post_training/"
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path)
    else:
        print(f"Directory already exists: {output_path}")


    model.eval()
    with torch.no_grad():
        for i, (inputs, _, filenames) in enumerate(test_loader):
            inputs = inputs.to(device)
            if args.model == "PNC_with_classification": outputs, _ = model(inputs)
            else: outputs = model(inputs)
            for j in range(inputs.size()[0]):
                output = outputs[j].permute(1, 2, 0).cpu().numpy() # outputs[j] original shape is (3, 224, 224), which need to convert to -> (224, 224, 3)
                # output = (output * 255).astype(np.uint8)

                # save the numpy array as image
                plt.imsave(os.path.join(output_path, filenames[j]), output)






