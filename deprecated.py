"""
autoencoder_train.py
"""
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

if __name__ == "__main__":
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



"""
models.py: 
"""
class PNC_256Unet_Autoencoder(nn.Module):
    def __init__(self):
        super(PNC_256Unet_Autoencoder, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)   # (3,256,256) -> (32,128,128)
        self.encoder2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # (32,128,128) -> (64,64,64)
        self.encoder3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (64,64,64) -> (128,32,32)
        self.encoder4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)# (128,32,32) -> (256,16,16)

        # Decoder
        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # (256,16,16) -> (128,32,32)
        self.decoder2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (128,32,32) -> (64,64,64)
        self.decoder3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # (64,64,64) -> (32,128,128)
        self.decoder4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # (32,128,128) -> (3,256,256)

        # Activation
        self.relu = nn.ReLU()

    def encode(self, x):
        """Encode the input image into latent representation."""
        x = self.relu(self.encoder1(x))  # (3,256,256) -> (32,128,128)
        x = self.relu(self.encoder2(x))  # (32,128,128) -> (64,64,64)
        x = self.relu(self.encoder3(x))  # (64,64,64) -> (128,32,32)
        x = self.relu(self.encoder4(x))  # (128,32,32) -> (256,16,16)
        return x

    def decode(self, x):
        """Decode the latent representation back to image space."""
        y1 = self.relu(self.decoder1(x))  # (256,16,16) -> (128,32,32)
        y2 = self.relu(self.decoder2(y1)) # (128,32,32) -> (64,64,64)
        y3 = self.relu(self.decoder3(y2)) # (64,64,64) -> (32,128,128)
        y4 = self.decoder4(y3)            # (32,128,128) -> (3,256,256)
        return torch.sigmoid(y4)  # Normalize output to [0, 1]

    def forward(self, x, tail_length=None):
        """Forward pass through the network."""
        x2 = self.encode(x)

        if tail_length is not None:
            # Zero out tail features for all samples in the batch
            batch_size, channels, _, _ = x2.size()
            tail_start = channels - tail_length
            x2 = x2.clone()
            x2[:, tail_start:, :, :] = 0

        reconstructed = self.decode(x2)
        return reconstructed

class PNC_with_classification(nn.Module):
    def __init__(self, autoencoder, num_classes):
        super(PNC_with_classification, self).__init__()

        # Use encoder from the trained autoencoder
        self.encoder = nn.Sequential(
            autoencoder.encoder1,  # (3, 224, 224) -> (16, 32, 32)
            autoencoder.encoder2   # (16, 32, 32) -> (10, 32, 32)
        )

        # Freeze encoder parameters (No updates during training)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 1x1 Conv for cross-channel learning + GAP to reduce dimensions
        self.feature_projection = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=1),   # (10, 32, 32) -> (32, 32, 32)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))        # (32, 32, 32) -> (32, 1, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():  # Freeze encoder
            features = self.encoder(x)  # (batch, 10, 32, 32)

        projected = self.feature_projection(features)  # (batch, 32, 1, 1)
        output = self.classifier(projected)
        return output

# PNC modified for random interspersed dropouts instead of tail dropouts
class PNC_Autoencoder_NoTail(nn.Module):
    def __init__(self):
        super(PNC_Autoencoder_NoTail, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (16, 32, 32)
        self.encoder2 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)  # (16, 32, 32) -> (10, 32, 32)

        # Decoder
        self.decoder1 = nn.ConvTranspose2d(10, 64, kernel_size=9, stride=7, padding=4, output_padding=6)  # (10, 32, 32) -> (64, 224, 224)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (64, 224, 224) -> (3, 224, 224)

        # Activation
        self.relu = nn.ReLU()

    def encode(self, x):
        """Perform encoding only."""
        x = self.relu(self.encoder1(x))  # (3, 224, 224) -> (16, 32, 32)
        x = self.relu(self.encoder2(x))  # (16, 32, 32) -> (10, 32, 32)
        return x

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.relu(self.decoder1(x))  # (10, 32, 32) -> (64, 224, 224)
        y2 = self.relu(self.decoder2(y1))  # (64, 224, 224) -> (64, 224, 224)
        y2 = y2 + y1  # Skip connection
        y3 = self.relu(self.decoder3(y2))  # (64, 224, 224) -> (64, 224, 224)
        y4 = self.relu(self.decoder3(y3))  # (64, 224, 224) -> (64, 224, 224)
        y4 = y4 + y3  # Skip connection
        y5 = self.final_layer(y4)  # (64, 224, 224) -> (3, 224, 224)
        y5 = torch.clamp(y5, min=0, max=1)  # Normalize output to [0, 1]
        return y5

    def forward(self, x, random_drop=None):
        # Encoder
        x1 = self.relu(self.encoder1(x))  # (3, 224, 224) -> (16, 32, 32)
        x2 = self.relu(self.encoder2(x1))  # (16, 32, 32) -> (10, 32, 32)

        if random_drop is not None: 
            # print("PNC_NoTail hit")
            # Zero out random interspersed features for all samples in the batch
            batch_size, channels, _, _ = x2.size()
            random_tail_length = random.randint(0, 9)
            tail_start = channels - random_tail_length
            x2 = x2.clone()  # Create a copy of the tensor to avoid in-place operations!
            if random_tail_length > 0:
                x2[:, tail_start:, :, :] = 0

        # Decoder
        y1 = self.relu(self.decoder1(x2))  # (10, 32, 32) -> (64, 224, 224)
        y2 = self.relu(self.decoder2(y1))  # (64, 224, 224) -> (64, 224, 224)
        y2 = y2 + y1  # Skip connection
        y3 = self.relu(self.decoder3(y2))  # (64, 224, 224) -> (64, 224, 224)
        y4 = self.relu(self.decoder3(y3))  # (64, 224, 224) -> (64, 224, 224)
        y4 = y4 + y3  # Skip connection
        y5 = self.final_layer(y4)  # (64, 224, 224) -> (3, 224, 224)
        y5 = torch.clamp(y5, min=0, max=1)  # Normalize output to [0, 1]

        return y5
    

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
