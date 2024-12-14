import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

classes = [
    'diving',
    'golf_front',
    'kick_front',
    'lifting',
    'riding_horse',
    'running',
    'skating',
    'swing_bench'
]

# Folder categories
folders = [
    "pnc_filled_0", "pnc_filled_10", "pnc_filled_20", "pnc_filled_30", "pnc_filled_40", "pnc_filled_50",
    "pnc_not_filled_0", "pnc_not_filled_10", "pnc_not_filled_20", "pnc_not_filled_30", "pnc_not_filled_40", "pnc_not_filled_50",
    "lrae_filled_0", "lrae_filled_10", "lrae_filled_20", "lrae_filled_30", "lrae_filled_40", "lrae_filled_50",
    "lrae_not_filled_0", "lrae_not_filled_10", "lrae_not_filled_20", "lrae_not_filled_30", "lrae_not_filled_40", "lrae_not_filled_50"
]

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [name for name in os.listdir(img_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load and convert image
        image = Image.open(img_path).convert("RGB")

        # Extract label index from filename
        try:
            label_idx = int(img_name.split('_')[1])  # Extract label index from filename
        except (IndexError, ValueError):
            raise ValueError(f"Invalid image filename format: {img_name}")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super(ComplexCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def calculate_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    # Load model
    model_path = 'best_model2.pth'  # Replace with your actual model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(classes)
    model = ComplexCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transforms
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Organize results by category
    results = {
        "PNC_filled": {"drop_rates": [], "accuracies": []},
        "PNC_not_filled": {"drop_rates": [], "accuracies": []},
        "LRAE_filled": {"drop_rates": [], "accuracies": []},
        "LRAE_not_filled": {"drop_rates": [], "accuracies": []}
    }

    # Test each folder
    for folder in folders:
        dataset = CustomImageDataset(img_dir=folder, transform=transform)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_accuracy = calculate_accuracy(test_dataloader, model, device)

        try: # Extract packet drop rate and category
            drop_rate = int(folder.split('_')[-1])  # Extract the number after the last '_'
        except ValueError:
            print(f"Invalid folder name format: {folder}")
            continue

        if "pnc_filled" in folder:
            category = "PNC_filled"
        elif "pnc_not_filled" in folder:
            category = "PNC_not_filled"
        elif "lrae_filled" in folder:
            category = "LRAE_filled"
        elif "lrae_not_filled" in folder:
            category = "LRAE_not_filled"
        else:
            print(f"Unknown folder category: {folder}")
            continue

        results[category]["drop_rates"].append(drop_rate)
        results[category]["accuracies"].append(test_accuracy)

        print(f"Folder {folder}, Category: {category}, Packet Drop Rate: {drop_rate}%, Test Accuracy: {test_accuracy:.2f}%")

    # Plotting results
    plt.figure(figsize=(10, 8))
    for category, data in results.items():
        plt.plot(data["drop_rates"], data["accuracies"], marker='o', label=category)

    plt.title('Classification Accuracy vs Packet Drop Rate')
    plt.xlabel('Packet Drop Rate (%)')
    plt.ylabel('Classification Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('classification_vs_packet_droprate_for_all_4_methods.png')
    plt.show()
