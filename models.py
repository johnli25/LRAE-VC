import torch
import torch.nn as nn
import random 

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



# TODO: FINALIZE LRAE-VC Architecture
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
    

class Compact_LRAE_VC_Autoencoder(nn.Module):
    def __init__(self):
        super(Compact_LRAE_VC_Autoencoder, self).__init__()

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
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # (128, 28, 28) -> (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 14, 14) -> (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 28, 28) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (64, 56, 56) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.decoder4 = nn.Sequential(
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
        x4 = self.encoder4(x3)  # (64, 14, 14)

        # Decoder
        y1 = self.decoder1(x4)  # (128, 28, 28)
        y1 = y1 + x3  # Skip connection

        y2 = self.decoder2(y1)  # (64, 56, 56)
        y2 = y2 + x2  # Skip connection

        y3 = self.decoder3(y2)  # (32, 112, 112)
        y3 = y3 + x1  # Skip connection

        y4 = self.decoder4(y3)  # (16, 224, 224)

        y5 = self.final_layer(self.dropout(y4))  # (3, 224, 224)
        y6 = self.sigmoid(y5)  # Normalize output to [0, 1]
        return y6



class LRAE_VC_Autoencoder_John(nn.Module):
    def __init__(self):
        super(Compact_LRAE_VC_Autoencoder, self).__init__()

        # Encoder: 3 layers
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # (3, 224, 224) -> (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (32, 112, 112) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 12, kernel_size=3, stride=2, padding=1),  # (64, 56, 56) -> (12, 30, 30)
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        # Decoder: 5 layers
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(12, 64, kernel_size=4, stride=2, padding=1),  # (12, 30, 30) -> (64, 60, 60)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (64, 60, 60) -> (32, 120, 120)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # (32, 120, 120) -> (32, 120, 120)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (32, 120, 120) -> (16, 240, 240)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),  # (16, 240, 240) -> (3, 240, 240)
        )

        # Activation for final output
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """Perform encoding only."""
        x = self.encoder1(x)  # (3, 224, 224) -> (32, 112, 112)
        x = self.encoder2(x)  # (32, 112, 112) -> (64, 56, 56)
        x = self.encoder3(x)  # (64, 56, 56) -> (12, 30, 30)
        return x

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.decoder1(x)  # (12, 30, 30) -> (64, 60, 60)
        y2 = self.decoder2(y1)  # (64, 60, 60) -> (32, 120, 120)
        y3 = self.decoder3(y2) + y2  # Skip connection within Decoder (32, 120, 120)
        y4 = self.decoder4(y3)  # (32, 120, 120) -> (16, 240, 240)
        y5 = self.decoder5(y4)  # (16, 240, 240) -> (3, 240, 240)
        y6 = self.sigmoid(y5)  # Normalize output to [0, 1]
        return y6

    def forward(self, x, random_drop=None):
        # Encoder
        x = self.encode(x)

        if random_drop:
            # Randomly drop tail channels
            random_tail_length = random.randint(0, 11)
            x = x.clone()  # Avoid in-place modification
            if random_tail_length > 0:
                x[:, -random_tail_length:, :, :] = 0

        # Decoder with localized skip connections
        y = self.decode(x)
        return y

    