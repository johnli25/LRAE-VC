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
from models import PNC_Autoencoder, PNC_256Unet_Autoencoder, PNC16, TestNew, TestNew2, TestNew3, PNC_with_classification, LRAE_VC_Autoencoder

# NOTE: uncomment below if you're using UCF Sports Action 
class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

test_img_names = {
    "diving_7", "diving_8", "golf_front_7", "golf_front_8", "kick_front_8", "kick_front_9",
    "lifting_5", "lifting_6", "riding_horse_8", "riding_horse_9", "running_7", "running_8",
    "running_9", "skating_8", "skating_9", "swing_bench_7", "swing_bench_8", "swing_bench_9"
}

# NOTE: uncomment below if you're using UCF101


def get_labels_from_filename(filenames):
    labels = []
    for filename in filenames:
        activity = "_".join(filename.split("_")[:-2])
        labels.append(class_map[activity]) # NOTE: class_map is global
    return labels


# Dataset class for loading images and ground truths
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Use the same image for both input and ground truth
        return image, image, self.img_names[idx]  # (image, same_image_as_ground_truth, img filename)


def train_autoencoder(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name, max_tail_length):
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
        val_loss = eval_autoencoder(model, val_loader, criterion, device, max_tail_length)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if max_tail_length: torch.save(model.state_dict(), f"{model_name}_best_validation_w_random_drops.pth")
            else: torch.save(model.state_dict(), f"{model_name}_best_validation_no_dropouts.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    plot_train_val_loss(train_losses, val_losses)

    # Save final model
    if max_tail_length: torch.save(model.state_dict(), f"{model_name}_final_w_random_drops.pth")
    else: torch.save(model.state_dict(), f"{model_name}_final_no_dropouts.pth")

    # Final Test: test_autoencoder()
    test_loss = eval_autoencoder(model, test_loader, criterion, device, max_tail_length)
    print(f"Final Test Loss: {test_loss:.4f}")


def eval_autoencoder(model, dataloader, criterion, device, max_tail_length):
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


def train_autoencoder_with_classification(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Train phase
        model.train()
        train_loss = 0
        for inputs, _, filenames in train_loader:
            labels = torch.tensor(get_labels_from_filename(filenames))

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        val_loss = test_autoencoder_with_classification(model, val_loader, device)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_name}_with_classification_best_validation.pth")
            print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved.")


    plot_train_val_loss(train_losses, val_losses)
    # Final Test
    accuracy = test_autoencoder_with_classification(model, test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    # Save final model
    torch.save(model.state_dict(), f"{model_name}_with_classification_final.pth")


def test_autoencoder_with_classification(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, _, filenames in dataloader:
            labels = torch.tensor(get_labels_from_filename(filenames))
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


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
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_256U", "PNC16", "TestNew", "TestNew2", "TestNew3", "PNC_NoTail", "PNC_with_classification", "LRAE_VC"], 
                            help="Model to train")
        return parser.parse_args()
    args = parse_args()

    ## Hyperparameters
    num_epochs = 28
    batch_size = 32
    learning_rate = 1e-3
    img_height, img_width = 224, 224 # NOTE: Dependent on autoencoder architecture!!
    path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/" # NOTE: already resized to 224x224 (so not really adaptable), but faster
    # path = "UCF_uncompressed_video_img_frames" # NOTE: more adaptable to different img dimensions because it's the original, BUT SLOWER!

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(path, transform=transform)

    test_indices = [
        i for i in range(len(dataset))
        if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
    ]
    train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

    # Split train_val_indices into train and validation
    np.random.shuffle(train_val_indices)
    train_size = int(0.9 * len(train_val_indices))
    val_size = len(train_val_indices) - train_size

    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # apply data augmentation to the train data
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform        
    test_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_tail_length = None
    if args.model == "PNC":
        model = PNC_Autoencoder().to(device)
        max_tail_length = 10 # true PNC

    if args.model == "PNC_256U":
        model = PNC_256Unet_Autoencoder().to(device)

    if args.model == "PNC16":
        model = PNC16().to(device)

    if args.model == "TestNew":
        model = TestNew().to(device)

    if args.model == "TestNew2":
        model = TestNew2().to(device)

    if args.model == "TestNew3":
        model = TestNew3().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    
    train_autoencoder(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, args.model, max_tail_length=max_tail_length) # max_tail_length = None or 10 (in the case of PNC)


    if args.model == "PNC_with_classification":
        model_autoencoder = PNC_Autoencoder().to(device)
        model_autoencoder.load_state_dict(torch.load("PNC_final_no_dropouts.pth"))
        model_autoencoder.eval()

        model_classifier = PNC_with_classification(model_autoencoder, num_classes=8).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_classifier.classifier.parameters(), lr=learning_rate)

        train_autoencoder_with_classification(
            model=model_classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            model_name="PNC_with_classification"
        )

    # Save images generated by DECODER ONLY! 
    output_path = "output_test_imgs_post_training/"
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path)

    if args.model == "PNC_with_classification":
        model_classifier.eval()  # Put classifier in eval mode
        model_autoencoder.eval()  # Put autoencoder in eval mode
        with torch.no_grad():
            correct, total = 0, 0
            for i, (inputs, _, filenames) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model_classifier(inputs)  # Forward pass through classifier

                _, predicted = torch.max(outputs, 1)

                ground_truth = get_labels_from_filename(filenames)
                ground_truth = torch.tensor(ground_truth).to(device)

                # Print predictions
                # for filename, pred, gt in zip(filenames, predicted.cpu().numpy(), ground_truth.cpu().numpy()):
                #     print(f"Frame: {filename}, Predicted Class: {pred}, Ground Truth: {gt}")
                correct += (predicted == ground_truth).sum().item() # Update accuracy
                total += ground_truth.size(0)

            # Final accuracy
            accuracy = 100 * correct / total if total > 0 else 0
            print(f"\nTotal Correct: {correct}/{total}")
            print(f"Classification Accuracy: {accuracy:.2f}%")

    else: # no classification! --> testing Reconstruction ONLY
        model.eval()  # Put the autoencoder in eval mode
        with torch.no_grad():
            for i, (inputs, _, filenames) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)  # Forward pass through autoencoder
                print("Outputs shape:", outputs.shape)

                # outputs is (batch_size, 3, image_h, image_w)
                # Save each reconstructed image
                for j in range(inputs.size(0)):
                    output_np = outputs[j].permute(1, 2, 0).cpu().numpy()  # (image_h, image_w, 3)
                    plt.imsave(os.path.join(output_path, filenames[j]), output_np)

