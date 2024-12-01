import os
import torch
import argparse
import socket
from torchvision import transforms
from PIL import Image
import struct
from models import PNC_Autoencoder, PNC_Autoencoder_with_Classification, LRAE_VC_Autoencoder, Compact_LRAE_VC_Autoencoder
import random

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sender Encoder")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory of images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained autoencoder model (.pth)")
    parser.add_argument("--host", type=str, required=True, help="Receiver IP address")
    parser.add_argument("--port", type=int, required=True, help="Receiver port number")
    return parser.parse_args()

def load_model(model_path, device):
    """Load the appropriate model based on the model path."""
    if "PNC" in model_path:
        model = PNC_Autoencoder()
    elif "PNC_with_Classification" in model_path:
        model = PNC_Autoencoder_with_Classification()
    elif "LRAE_VC" in model_path:
        model = LRAE_VC_Autoencoder()
    elif "Compact_LRAE_VC" in model_path:
        model = Compact_LRAE_VC_Autoencoder()
    else:
        raise ValueError(f"Unknown model type in model_path: {model_path}")
    
    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def encode_and_send(input_dir, model, host, port, device):
    """Encode images from the input directory and send them over the network."""
    # Prepare socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    test_dataset = {
        "diving_7", "diving_8", "golf_front_7", "golf_front_8", "kick_front_8", "kick_front_9",
        "lifting_5", "lifting_6", "riding_horse_8", "riding_horse_9", "running_7", "running_8",
        "running_9", "skating_8", "skating_9", "swing_bench_7", "swing_bench_8", "swing_bench_9"
    }
    
    try:
        last_video_identifier = None
        video_number = -1
        image_number = 0
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            video_identifier = '_'.join(filename.split('_')[:-1])
            if video_identifier not in test_dataset:
                continue 
            if video_identifier != last_video_identifier:
                video_number += 1
                last_video_identifier = video_identifier
                image_number = 0 # reset image number for new video

            image = Image.open(file_path).convert("RGB")
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            with torch.no_grad():
                encoded_features = model.encode(image_tensor)  # shape = (1, 10, 32, 32)
            for feature_num in range(encoded_features.shape[1]): # encoded_features.shape[1] = number of features in model
                if random.random() <= 0.25: # NOTE: 20% chance of feature/packet drop/loss, uncommented this
                    print("Dropping packet for filename, video_number, image_number, feature_num:", filename, video_number, image_number, feature_num)
                    continue
                # Extract the specific feature
                feature = encoded_features[0, feature_num, :, :].cpu()  # shape = (1, 32, 32)
                
                # Metadata for transmission
                metadata = (video_number, image_number, feature_num)  # Tuple of metadata
                
                # Serialize metadata and features
                metadata_bytes = struct.pack("III", *metadata)  # Pack as 3 unsigned integers
                feature_bytes = feature.numpy().tobytes()  # Serialize as binary
                
                # Send size of the entire data block
                sock.sendall(struct.pack("I", len(metadata_bytes) + len(feature_bytes)))
                
                # Send metadata and feature bytes
                sock.sendall(metadata_bytes + feature_bytes)
            
            image_number += 1
    finally:
        sock.close()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = load_model(args.model_path, device)
    encode_and_send(args.input_dir, model, args.host, args.port, device)