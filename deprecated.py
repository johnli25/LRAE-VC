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

    def forward(self, x, tail_length=None):
        # Encoder
        x1 = self.relu(self.encoder1(x))  # (3, 224, 224) -> (16, 32, 32)
        x2 = self.relu(self.encoder2(x1))  # (16, 32, 32) -> (10, 32, 32)

        if tail_length is not None:
            # Zero out tail features for all samples in the batch
            batch_size, channels, _, _ = x2.size()
            tail_start = channels - tail_length
            x2 = x2.clone()  # Create a copy of the tensor to avoid in-place operations!
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
    

class TestNew(nn.Module):
    def __init__(self):
        super(TestNew, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (3, 224, 224) -> (16, 112, 112)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # (16, 112, 112) -> (24, 56, 56)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1),  # (24, 56, 56) -> (24, 48, 48)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(24, 48, kernel_size=3, stride=2, padding=1, output_padding=1),  # (24, 48, 48) -> (48, 56, 56)
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1)
        )

        self.residual1 = nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0)  # (48, 56, 56) -> (48, 56, 56) 

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (48, 56, 56) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.residual2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)  # (32, 112, 112) -> (32, 112, 112)  

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 112, 112) -> (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.residual3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # (64, 224, 224) -> (64, 224, 224)

        # Added additional refinement layers (64, 224, 224) -> (64, 224, 224)
        self.decoder4_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.decoder4_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        # Final output layer
        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (64, 224, 224) -> (3, 224, 224)
            nn.Sigmoid()
        )

        # Regularization
        self.dropout = nn.Dropout(0.3)  

    def encode(self, x):
        """Perform encoding only."""
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x  # Returns a latent representation of (24, 48, 48)

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.decoder1(x)
        y1 = y1 + self.residual1(y1)

        y2 = self.decoder2(y1)
        y2 = y2 + self.residual2(y2)

        y3 = self.decoder3(y2)
        y3 = y3 + self.residual3(y3)

        # Additional Refinement Layers
        y4 = self.decoder4_1(y3)
        y4 = self.decoder4_2(y4)

        y5 = self.decoder5(y4)
        return y5

    def forward(self, x, random_drop=None): 
        latent = self.encode(x)

        if random_drop is not None:
            print("hit LRAE_VC random_drop") 
            mask = (torch.rand(latent.size(1)) > random_drop).float().to(latent.device)
            mask = mask.view(1, -1, 1, 1)
            latent = latent * mask

        output = self.decode(latent)
        return output


class TestNew2(nn.Module):
    def __init__(self):
        super(TestNew2, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (3, 224, 224) -> (16, 112, 112)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # (16, 112, 112) -> (24, 56, 56)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(24, 48, kernel_size=3, stride=2, padding=1, output_padding=1),  # (24, 56, 56) -> (48, 112, 112)
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1)
        )

        self.residual1 = nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0)  # (48, 112, 112) -> (48, 112, 112)  

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (48, 112, 112) -> (32, 224, 224)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.residual2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)  # (32, 224, 224) -> (32, 224, 224)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 224, 224) -> (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.residual3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # (64, 224, 224) -> (64, 224, 224)

        # Refinement layers (64, 224, 224) -> (64, 224, 224)
        self.decoder4_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.decoder4_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        # Final output layer
        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (64, 224, 224) -> (3, 224, 224)
            nn.Sigmoid()
        )

        # Regularization
        self.dropout = nn.Dropout(0.3)

    def encode(self, x):
        """Perform encoding only."""
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x  # Returns a latent representation of (24, 56, 56)

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.decoder1(x)
        y1 = y1 + self.residual1(y1)

        y2 = self.decoder2(y1)
        y2 = y2 + self.residual2(y2)

        y3 = self.decoder3(y2)
        y3 = y3 + self.residual3(y3)

        # Additional Refinement Layers
        y4 = self.decoder4_1(y3)
        y4 = self.decoder4_2(y4)

        y5 = self.decoder5(y4)
        return y5

    def forward(self, x, random_drop=None):
        latent = self.encode(x)

        if random_drop is not None:
            print("hit LRAE_VC random_drop")
            mask = (torch.rand(latent.size(1)) > random_drop).float().to(latent.device)
            mask = mask.view(1, -1, 1, 1)
            latent = latent * mask

        output = self.decode(latent)
        return output
    

class TestNew3(nn.Module):
    def __init__(self):
        super(TestNew3, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (3,224,224) -> (16,112,112)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        # Modified encoder2 with stride=3 for stronger downsampling:
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=3, padding=1),  # (16,112,112) -> (24,38,38)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        # Modified decoder1 to match encoder2's stride:
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(24, 48, kernel_size=3, stride=3, padding=1, output_padding=0),  # (24,38,38) -> (48,112,112)
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1)
        )

        self.residual1 = nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0)  # (48,112,112) -> (48,112,112)

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (48,112,112) -> (32,224,224)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.residual2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)  # (32,224,224) -> (32,224,224)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32,224,224) -> (64,224,224)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.residual3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # (64,224,224) -> (64,224,224)

        # Refinement layers
        self.decoder4_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.decoder4_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        # Final output layer
        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (64,224,224) -> (3,224,224)
            nn.Sigmoid()
        )

        # Regularization
        self.dropout = nn.Dropout(0.3)

    def encode(self, x):
        """Perform encoding only."""
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x  # Returns latent of shape (24,38,38)

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.decoder1(x)
        y1 = y1 + self.residual1(y1)

        y2 = self.decoder2(y1)
        y2 = y2 + self.residual2(y2)

        y3 = self.decoder3(y2)
        y3 = y3 + self.residual3(y3)

        # Refinement Layers
        y4 = self.decoder4_1(y3)
        y4 = self.decoder4_2(y4)

        y5 = self.decoder5(y4)
        return y5

    def forward(self, x, random_drop=None):
        latent = self.encode(x)

        if random_drop is not None:
            print("hit random_drop")
            mask = (torch.rand(latent.size(1)) > random_drop).float().to(latent.device)
            mask = mask.view(1, -1, 1, 1)
            latent = latent * mask

        latent = self.dropout(latent)
        output = self.decode(latent)
        return output


class LRAE_VC_Autoencoder(nn.Module):
    def __init__(self):
        super(LRAE_VC_Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),  # (3, 224, 224) -> (8, 112, 112)
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=4, padding=1),  # (8, 112, 112) -> (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16,32, kernel_size=3, stride=4, padding=1, output_padding=3),  # (16, 28, 28) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.residual1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)  # (32, 112, 112) -> (32, 112, 112) 

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 112, 112) -> (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.residual2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # (64, 224, 224) -> (64, 224, 224)  

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (64, 224, 224) -> (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.residual3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # (64, 224, 224) -> (64, 224, 224)

        self.decoder4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # (64, 224, 224) -> (3, 224, 224)
            nn.Sigmoid()
        )

        # Regularization
        self.dropout = nn.Dropout(0.3) 

    def encode(self, x):
        """Perform encoding only."""
        x = self.encoder1(x)
        x = self.encoder2(x)
        return x

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.decoder1(x)
        y1 = y1 + self.residual1(y1)

        y2 = self.decoder2(y1)
        y2 = y2 + self.residual2(y2)

        y3 = self.decoder3(y2)
        y3 = y3 + self.residual3(y3)

        # y3 = self.dropout(y3)  # NOTE: Optional, so comment out if you want

        y4 = self.decoder4(y3)
        return y4

    def forward(self, x, random_drop=None): 
        latent = self.encode(x)

        if random_drop is not None:
            print("hit LRAE_VC random_drop") 
            mask = (torch.rand(latent.size(1)) > random_drop).float().to(latent.device)

            mask = mask.view(1, -1, 1, 1)

            latent = latent * mask

        output = self.decode(latent)
        return output

class FrameSequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FrameSequenceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=self.bidirectional)

        # Fully connected layer to project hidden state to output
        #self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass of the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, 32, 32].

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Reshape input from [batch_size, sequence_length, 32, 32] -> [batch_size, sequence_length, 1024]
        batch_size, sequence_length, height, width = x.shape
        
        x = x.view(batch_size, sequence_length, -1)  # Flatten spatial dimensions
        
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply fully connected layer to each timestep
        output = self.fc(lstm_out)
        
        # Reshape output back to [batch_size, sequence_length, 32, 32]
        output = output.view(batch_size, sequence_length, height, width)
        return output
    
