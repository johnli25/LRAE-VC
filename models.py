import torch
import torch.nn as nn
import numpy as np
import random 
import math 

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


class PNC16(nn.Module):
    def __init__(self):
        super(PNC16, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (16, 32, 32)
        self.encoder2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  # (16, 32, 32) -> (16, 32, 32)

        # Decoder
        self.decoder1 = nn.ConvTranspose2d(16, 64, kernel_size=9, stride=7, padding=4, output_padding=6)  # (16, 32, 32) -> (64, 224, 224)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (64, 224, 224) -> (3, 224, 224)

        # Activation
        self.relu = nn.ReLU()

    def encode(self, x):
        """Perform encoding only."""
        x = self.relu(self.encoder1(x))  # (3, 224, 224) -> (16, 32, 32)
        x = self.relu(self.encoder2(x))  # (16, 32, 32) -> (16, 32, 32)
        return x

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.relu(self.decoder1(x))  # (16, 32, 32) -> (64, 224, 224)
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
        x2 = self.encode(x)  # (3, 224, 224) -> (16, 32, 32)
        if tail_length is not None:
            # Zero out tail features for all samples in the batch
            batch_size, channels, _, _ = x2.size()
            tail_start = channels - tail_length
            print(f"tail_len = {tail_length}; tail_start = {tail_start}")
            x2 = x2.clone()  # Create a copy of the tensor to avoid in-place operations!
            x2[:, tail_start:, :, :] = 0

        # Decoder
        y5 = self.decode(x2)  # (16, 32, 32) -> (3, 224, 224)

        return y5
    
class PNC16Encoder(nn.Module): # Conv Encoder
    def __init__(self):
        super(PNC16Encoder, self).__init__()
        
        # Encoder layers exactly matching PNC16
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (16, 32, 32)
        self.encoder2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  # (16, 32, 32) -> (16, 32, 32)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """Encodes the input image to (16, 32, 32) feature space."""
        x = self.relu(self.encoder1(x))  # (3, 224, 224) -> (16, 32, 32)
        x = self.relu(self.encoder2(x))  # (16, 32, 32) -> (16, 32, 32)
        return x

class PNC16Decoder(nn.Module): # Conv Decoder
    def __init__(self):
        super().__init__()
        # Same conv layers as in PNC16 decode
        self.decoder1 = nn.ConvTranspose2d(16, 64, kernel_size=9, stride=7, padding=4, output_padding=6)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch_size, 16, 32, 32)
        y1 = self.relu(self.decoder1(x))   # -> (batch_size, 64, 224, 224)
        y2 = self.relu(self.decoder2(y1))    # -> (batch_size, 64, 224, 224)
        y2 = y2 + y1
        y3 = self.relu(self.decoder3(y2))    # -> (batch_size, 64, 224, 224)
        y4 = self.relu(self.decoder3(y3))    # -> (batch_size, 64, 224, 224)
        y4 = y4 + y3
        y5 = self.final_layer(y4)            # -> (batch_size, 3, 224, 224)
        y5 = torch.clamp(y5, 0, 1)
        return y5
    
    
class PNC32(nn.Module):
    def __init__(self):
        super(PNC32, self).__init__()
        
        # Encoder: change number of channels from 16 to 32.
        self.encoder1 = nn.Conv2d(3, 32, kernel_size=9, stride=7, padding=4)  # (3, 224, 224) -> (32, 32, 32)
        self.encoder2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # (32, 32, 32) -> (32, 32, 32)

        # Decoder: update first decoder layer to accept 32 channels.
        self.decoder1 = nn.ConvTranspose2d(32, 64, kernel_size=9, stride=7, padding=4, output_padding=6)  # (32, 32, 32) -> (64, 224, 224)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # (64, 224, 224) -> (64, 224, 224)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (64, 224, 224) -> (3, 224, 224)

        self.relu = nn.ReLU()

    def encode(self, x):
        """Perform encoding only."""
        x = self.relu(self.encoder1(x))  # (3, 224, 224) -> (32, 32, 32)
        x = self.relu(self.encoder2(x))  # (32, 32, 32) -> (32, 32, 32)
        return x

    def decode(self, x):
        """Perform decoding only."""
        y1 = self.relu(self.decoder1(x))  # (32, 32, 32) -> (64, 224, 224)
        y2 = self.relu(self.decoder2(y1))  # (64, 224, 224) -> (64, 224, 224)
        y2 = y2 + y1  # Skip connection
        y3 = self.relu(self.decoder3(y2))  # (64, 224, 224) -> (64, 224, 224)
        # Example: using decoder3 a second time for another pass (you might choose to define a separate layer)
        y4 = self.relu(self.decoder3(y3))  # (64, 224, 224) -> (64, 224, 224)
        y4 = y4 + y3  # Skip connection
        y5 = self.final_layer(y4)  # (64, 224, 224) -> (3, 224, 224)
        y5 = torch.clamp(y5, min=0, max=1)  # Normalize output to [0, 1]
        return y5

    def forward(self, x, tail_length=None, quantize_level=False):
        # Encoder
        x2 = self.encode(x)  # (3, 224, 224) -> (32, 32, 32)
        if tail_length is not None:
            batch_size, channels, _, _ = x2.size()
            tail_start = channels - tail_length
            # print(f"tail_len = {tail_length}; tail_start = {tail_start}")
            x2 = x2.clone()  # Avoid in-place modification
            x2[:, tail_start:, :, :] = 0

        # Quantization: simulate 8-bit quantization on the latent
        if quantize_level > 0:
            # print("quantize_level in PNC32:", quantize_level)
            x2 = self.quantize(x2, levels=quantize_level) # NOTE: levels=256 still maintains range [0,1] for x2; it just quantizes the values to 256 levels.

        # Decoder
        y5 = self.decode(x2)  # (32, 32, 32) -> (3, 224, 224)
        return y5
    
    def quantize(self, x, levels=256):
        """
        Simulate quantization by clamping x to [0,1] and then rounding to levels-1 steps.
        For optimal reconstruction quality, you should consider using quantization-aware training.
        """
        # For simplicity, assume x is roughly in [0, 1]. Otherwise, consider a learned scale.
        x_clamped = torch.clamp(x, 0, 1)
        x_quantized = torch.round(x_clamped * (levels - 1)) / (levels - 1)
        return x_quantized
    

class PNC32Encoder(nn.Module):
    """
    Encoder that maps an input image (3, 224, 224) to a latent space with 32 channels 
    and spatial size 32x32, using the architecture of PNC32.
    """
    def __init__(self):
        super(PNC32Encoder, self).__init__()
        self.encoder1 = nn.Conv2d(3, 32, kernel_size=9, stride=7, padding=4)   # (3, 224, 224) -> (32, 32, 32)
        self.encoder2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)   # (32, 32, 32) -> (32, 32, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.encoder1(x))
        x = self.relu(self.encoder2(x))
        return x

class PNC32Decoder(nn.Module):
    """
    Decoder that reconstructs an image from a 32-channel latent representation with spatial
    dimensions 32x32 to an output image (3, 224, 224) using skip connections.
    """
    def __init__(self):
        super(PNC32Decoder, self).__init__()
        self.decoder1 = nn.ConvTranspose2d(32, 64, kernel_size=9, stride=7, padding=4, output_padding=6)  # (32, 32, 32) -> (64, 224, 224)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)                               # (64, 224, 224) -> (64, 224, 224)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)                               # (64, 224, 224) -> (64, 224, 224)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)                             # (64, 224, 224) -> (3, 224, 224)
        self.relu = nn.ReLU()

    def forward(self, x):
        y1 = self.relu(self.decoder1(x))        # -> (64, 224, 224)
        y2 = self.relu(self.decoder2(y1))         # -> (64, 224, 224)
        y2 = y2 + y1                            # Skip connection
        y3 = self.relu(self.decoder3(y2))         # -> (64, 224, 224)
        y4 = self.relu(self.decoder3(y3))         # Another pass through decoder3
        y4 = y4 + y3                            # Skip connection
        y5 = self.final_layer(y4)               # -> (3, 224, 224)
        y5 = torch.clamp(y5, min=0, max=1)        # Clamp output to [0, 1]
        return y5


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels= 4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c, quality=1.0): 
        """
        x: (batch, input_channels, H, W) - input tensor
        h: (batch, hidden_channels, H, W) - hidden state
        c: (batch, hidden_channels, H, W) - cell state
        """
        combined = torch.cat([x, h], dim=1)  # Concatenate along channel dimension
        gates = self.conv(combined)  # (batch, 4 * hidden_channels, H, W)

        # Split the gates into input, forget, cell, and output gates
        chunk = self.hidden_channels
        i = torch.sigmoid(gates[:, 0:chunk]) # input gate determines how much of the new input to --> cell state
        f = torch.sigmoid(gates[:, chunk:2*chunk]) # Decides what information to discard from the previous cell state.
        o = torch.sigmoid(gates[:, 2*chunk:3*chunk]) # Controls how much of the cell state should be exposed as the hidden state (output) for current time step.
        g = torch.tanh(gates[:, 3*chunk:4*chunk]) # Candidate Cell State: Represents the new info that could be added to the cell state.

        # Update cell state and hidden state
        c = f * c + i * g 
        h = o * torch.tanh(c) # # Hidden State: acts as output of the LSTM cell. It is used both to produce the cell’s final output and to serve as part of the input to the next time step.
        
        return h, c
    

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)
        self.hidden_channels = hidden_channels
    
    def forward(self, x_seq):
        """
        x_seq: (batch, seq_len, input_channels, H, W)
        drop_levels: list of length seq_len, each element is a list of length batch_size
        returns: (batch, seq_len, hidden_channels, H, W)
        """
        bsz, seq_len, _, H, W = x_seq.shape 
        # initialize hidden and cell states to zeros before processing the sequence
        h = torch.zeros(bsz, self.hidden_channels, H, W).to(x_seq.device)
        c = torch.zeros_like(h)

        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t] # extracts the t-th frame, where shape is (batch, input_channels, H, W) e.g. (batch, 32, 32, 32)
            h, c = self.cell(x_t, h, c)  # update hidden and cell states
            outputs.append(h.unsqueeze(1)) # NOTE: append the hidden state for this time step. unsqueeze(1) b/c we want to add a new dimension for time!!

        return torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_channels, H, W)
    
class ConvBiLSTM(nn.Module):
    """
    Truly bidirectional ConvLSTM that processes the whole sequence forward and backward.
    we combine the forward and backward hidden states via concatenation at each time step.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels 

        # forward and backward cells
        self.forward_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.backward_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x_seq):
        """
        x: (batch, seq_len, input_channels, H, W)
        returns h_out of shape (batch, seq_len, 2 * hidden_channels, H, W) since forward + backward states are concatenated
        """
        batch_size, seq_len, input_channels, H, W = x_seq.shape 

        # buffers for forward pass
        h_forward = []

        # initialize
        h_f = torch.zeros(batch_size, self.hidden_channels, H, W).to(x_seq.device)
        c_f = torch.zeros_like(h_f)

        for t in range(seq_len):
            x_t = x_seq[:, t]
            h_f, c_f = self.forward_cell(x_t, h_f, c_f)
            h_forward.append(h_f.unsqueeze(1))

        # buffers for backward pass
        h_backward = []

        # initialize
        h_b = torch.zeros(batch_size, self.hidden_channels, H, W).to(x_seq.device)
        c_b = torch.zeros_like(h_b)

        for t in reversed(range(seq_len)):
            x_t = x_seq[:, t]
            h_b, c_b = self.backward_cell(x_t, h_b, c_b)
            h_backward.append(h_b.unsqueeze(1))

        # concatenate forward and backward hidden states
        outputs = []
        for t in range(seq_len):
            h_t = h_forward[t]  # (batch, hidden_channels, H, W)
            h_t_b = h_backward[t]  # (batch, hidden_channels, H, W)
            h_t_out = torch.cat([h_t, h_t_b], dim=1)  # (batch, 2 * hidden_channels, H, W)
            outputs.append(h_t_out)

        return torch.cat(outputs, dim=1)  # (batch, seq_len, 2 * hidden_channels, H, W)
    

class ConvLSTM_AE(nn.Module): # NOTE: this does "automatic/default" 0 padding for feature/channel dropouts
    def __init__(self, total_channels, hidden_channels, ae_model_name, bidirectional=False):
        super().__init__()
        self.total_channels = total_channels
        self.hidden_channels = hidden_channels
        self.bidirectional = bidirectional

        # 1) encoder
        if ae_model_name == "PNC16":
            print("Using PNC16 Encoder")
            self.encoder = PNC16Encoder()
        elif ae_model_name == "PNC32":
            print("Using PNC32 Encoder")    
            self.encoder = PNC32Encoder()

        #2 Feed forward zero-padded partial latents to LSTM. LSTM sees input_channels=total_channels 
        lstm_out_channels = 2 * hidden_channels if bidirectional else hidden_channels

        if bidirectional:
            self.conv_lstm = ConvBiLSTM(input_channels=total_channels, hidden_channels=hidden_channels)
        else:
            self.conv_lstm = ConvLSTM(input_channels=total_channels, hidden_channels=hidden_channels)

        print("lstm_out_channels:", lstm_out_channels)
        # 3) If LSTM's hidden state dimensions/channels != total_channels, map LSTM's hidden state channels to total_channels (which is input to decoder) e.g. if hidden_channels=32, total_channels=16 for PNC16
        if lstm_out_channels != total_channels:
            self.project_lstm_to_latent = nn.Conv2d(lstm_out_channels, total_channels, kernel_size=1, stride=1, padding=0)
        else: 
            self.project_lstm_to_latent = None
        
        # 5) finally, decoder
        if ae_model_name == "PNC16":
            print("Using PNC16 Decoder")
            self.decoder = PNC16Decoder()
        elif ae_model_name == "PNC32":
            print("Using PNC32 Decoder")
            self.decoder = PNC32Decoder()

    def forward(self, x_seq, drop=0, eval_random=False, eval_consecutive=None, quantize=False):
        """
        x_seq: (batch_size, seq_len, 3, 224, 224)   
        returns (batch_size, seq_len, 3, 224, 224) reconstructed video frames/imgs sequence
        """
        # print(self.conv_lstm, self.project_lstm_to_latent, self.decoder)
        bsz, seq_len, c, h, w = x_seq.shape
        drop_levels = []
        # 1) Encode + randomly drop channels
        partial_list = []
        for t in range(seq_len): # So outer loop needs to loop thru each time step (of frames) in the sequence
            frame = x_seq[:, t] # needs to be of shape (batch, 3, 224, 224), so that...
            features = self.encoder(frame) # .encoder(frame) returns (batch, feature_maps, height, width)!
            current_drop = []
        
            # 2) Randomly drop tail channels/features
            if drop > 0:
                features = features.clone()  # avoid in-place modifications
                # use consecutive dropout if specified
                if eval_consecutive is not None:
                    # NOTE: Using below code, the first frame of each cycle (when t % (eval_consecutive+1) drops 0 features, and the subsequent eval_consecutive frames drop 'drop' features) 
                    consecutive_drops = (
                        torch.zeros((features.size(0),), device=features.device)
                        if t % (eval_consecutive + 1) != 0 # or t == 0 # add t == 0 if you want to unconditionally force the first frame to have NO loss/dropout
                        else torch.full((features.size(0),), drop, device=features.device)
                    )

                    for i, consecutive_drop in enumerate(consecutive_drops):
                        current_drop.append(consecutive_drop.item())
                        if consecutive_drop > 0:
                            features[i, -consecutive_drop:, :, :] = 0.0

                # otherwise use random dropout logic (in training or eval_random mode)    
                elif self.training or eval_random:
                    # Training: randomly drop 0 to `drop` tail channels per sample.
                    random_drops = torch.randint(low=0, high=drop + 1, size=(features.size(0),))
                    for i, random_drop in enumerate(random_drops):
                        current_drop.append(random_drop.item())
                        if random_drop > 0:
                            features[i, -random_drop:, :, :] = 0.0
                else:
                    # Evaluation: drop a CONSTANT number of tail channels (e.g., exactly 'drop' amt of channels)
                    features[:, -drop:, :, :] = 0.0

            if quantize > 0:
                features = self.quantize(features, levels=quantize)

            if self.training or eval_random or eval_consecutive != None: # if eval_real, append drop levels for each sample
                drop_levels.append(current_drop)
            partial_list.append(features) # (batch, 16, 32, 32)

        # Convert drop_levels (a list of length seq_len, each an array of length bsz) to shape (bsz, seq_len) by transposing the list:
        drop_levels = list(map(list, zip(*drop_levels)))
        drop_levels_tensor = torch.tensor(drop_levels, device=x_seq.device)
        # print("drop_levels:", drop_levels_tensor.shape)

        # stack features along the time dimension (seq_len dimension = 1)
        lstm_input = torch.stack(partial_list, dim=1) # (batch, seq_len, 16, 32, 32)

        lstm_out = self.conv_lstm(x_seq=lstm_input) # (batch, seq_len, hidden_channels, 32, 32) 

        # if needed, map hidden state to total_channels, and finally decode!!
        if self.project_lstm_to_latent is not None:
            # Flatten batch and time dimensions for mapping
            lstm_channels = 2 * self.hidden_channels if self.bidirectional else self.hidden_channels
            imputed_latents = self.project_lstm_to_latent(lstm_out.view(-1, lstm_channels, 32, 32))
            # Reshape back to (batch, seq_len, total_channels, 32, 32)
            imputed_latents = imputed_latents.view(bsz, seq_len, self.total_channels, 32, 32) # Example: 32 hidden channels -> 16 total channels (PNC16 AE)
        else:
            imputed_latents = lstm_out  # if no mapping is needed
        
        # decode each time step
        outputs = []
        for t in range(seq_len):
            h_t = imputed_latents[:, t]  # (batch, total_channels, 32, 32)
            recon_frame = self.decoder(h_t)  # (batch, 3, 224, 224)
            outputs.append(recon_frame.unsqueeze(1))
        
        recon = torch.cat(outputs, dim=1)  # (batch, seq_len, 3, 224, 224)

        if eval_random or eval_consecutive != None:
            return recon, imputed_latents, drop_levels_tensor
        else:
            return recon, imputed_latents, None
        
    def quantize(self, x, levels=256):  
        """
        Simulate quantization by clamping x to [0,1] and then rounding to levels-1 steps.
        For optimal reconstruction quality, you should consider using quantization-aware training.
        """
        # For simplicity, assume x is roughly in [0, 1]. Otherwise, consider a learned scale.
        x_clamped = torch.clamp(x, 0, 1)
        x_quantized = torch.round(x_clamped * (levels - 1)) / (levels - 1)
        return x_quantized


    
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
    
