import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import (PNC_Autoencoder, PNC_Autoencoder_NoTail, LRAE_VC_Autoencoder)


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


# for PNC and tail dropouts
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
            tail_len = torch.randint(0, max_tail_length + 1, (1,)).item() if max_tail_length else None
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, tail_len) 
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
            if max_tail_length: torch.save(model.state_dict(), f"{model_name}_best_validation_w_random_drops.pth")
            else: torch.save(model.state_dict(), f"{model_name}_best_validation_no_dropouts.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Final Test
    test_loss = test_autoencoder(model, test_loader, criterion, device, max_tail_length)
    print(f"Final Test Loss: {test_loss:.4f}")

    plot_train_val_loss(train_losses, val_losses)

    # Save final model
    if max_tail_length: torch.save(model.state_dict(), f"{model_name}_final_w_random_drops.pth")
    else: torch.save(model.state_dict(), f"{model_name}_final_no_dropouts.pth")


def test_autoencoder(model, dataloader, criterion, device, max_tail_length):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            torch.manual_seed(seed=42)
            tail_len = torch.randint(0, max_tail_length + 1, (1,)).item() if max_tail_length else None
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, tail_len) 
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    return test_loss / len(dataloader.dataset)



# for LRAE-VC and interspersed dropouts
def train_autoencoder_LRAE(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, random_drop):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, random_drop) 
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        val_loss = test_autoencoder_LRAE(model, val_loader, criterion, device, random_drop)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if random_drop: torch.save(model.state_dict(), f"{model_name}_best_validation_w_random_drops.pth")
            else: torch.save(model.state_dict(), f"{model_name}_best_validation_no_dropouts.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Final Test
    test_loss = test_autoencoder_LRAE(model, test_loader, criterion, device, random_drop)
    print(f"Final Test Loss: {test_loss:.4f}")

    plot_train_val_loss(train_losses, val_losses)

    # Save final model
    if random_drop: torch.save(model.state_dict(), f"{model_name}_final_w_random_drops.pth")
    else: torch.save(model.state_dict(), f"{model_name}_final_no_dropouts.pth")


def test_autoencoder_LRAE(model, dataloader, criterion, device, random_drop):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, random_drop) 
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

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
    parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_NoTail", "PNC_with_classification", "LRAE_VC"], 
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
    test_img_names = {
        "diving_7", "diving_8", "golf_front_7", "golf_front_8", "kick_front_8", "kick_front_9",
        "lifting_5", "lifting_6", "riding_horse_8", "riding_horse_9", "running_7", "running_8",
        "running_9", "skating_8", "skating_9", "swing_bench_7", "swing_bench_8", "swing_bench_9"
    }

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
        train_autoencoder(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, args.model, None) # NOTE: set tail_length to None to use the full sequence (0 features dropped) OR set tail_length=tail_length to enable stochastic tail-dropout

    if args.model == "LRAE_VC":
        model = LRAE_VC_Autoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_autoencoder_LRAE(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, args.model, None) # I set drop probability to 0.14, but you can change it to whatever you want

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
