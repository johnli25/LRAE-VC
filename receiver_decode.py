import os
import torch
import socket
import argparse
import struct
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from models import PNC_Autoencoder, PNC_Autoencoder_with_Classification, LRAE_VC_Autoencoder, Compact_LRAE_VC_Autoencoder, FrameSequenceLSTM

frameID_to_latent_encodings = defaultdict(lambda: np.zeros((10, 32, 32), dtype=np.float32))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Receiver Decoder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained autoencoder model (.pth)")
    parser.add_argument("--host", type=str, required=True, help="Host IP address to bind the server")
    parser.add_argument("--port", type=int, required=True, help="Port number to bind the server")
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


def decode_and_store(conn):
    """Decode received encoded features and store latent encodings."""
    while True:
        try:
            # Receive the size of the incoming data
            size_data = conn.recv(4)
            if not size_data:  # End of transmission when no data is received
                break
            data_size = struct.unpack("I", size_data)[0]
            
            # Receive metadata and feature bytes
            data = conn.recv(data_size)
            metadata_bytes = data[:12]  # First 12 bytes are metadata (3 unsigned integers) (vid #, img #, feature #)
            feature_bytes = data[12:]  # Remaining bytes are the feature
            
            # Deserialize metadata
            video_number, image_number, feature_number = struct.unpack("III", metadata_bytes)
            
            # Deserialize the feature
            feature = np.frombuffer(feature_bytes, dtype=np.float32).reshape(32, 32)
            
            # Update the latent encoding array
            frameID_to_latent_encodings[(video_number, image_number)][feature_number, :, :] = feature # assumes 1 feature per frame (no frames entirely dropped)
            # print(f"Updated latent encoding for video {video_number}, image {image_number}, feature {feature_number}")
        except Exception as e:
            print(f"Error during decoding or storing: {e}")
            break

# AJ FILLS THIS IN... Need to fill in missing features in the dictionary using the corresponding LSTM model
def feature_filler(device, input_dim, hidden_dim, output_dim, num_layers):
    video_tensors = {}
    for (video_number, image_number), tensor in frameID_to_latent_encodings.items():
        if video_number not in video_tensors:
            video_tensors[video_number] = []
        video_tensors[video_number].append((image_number, tensor))

    # Sort image_numbers and stack tensors for each video_number
    for video_number, tensors in video_tensors.items():
        # Sort tensors by image_number
        tensors.sort(key=lambda x: x[0])
        # Stack tensors
        video_tensor = np.stack([tensor for _, tensor in tensors])
        video_tensor = torch.tensor(video_tensor, dtype=torch.float32)
        
        video_tensor = video_tensor.transpose(0, 1).to(device) # Now (10, X, 32, 32)

        for feature_num in range(video_tensor.shape[0]):
            for frame_num in range(video_tensor.shape[1]):
                if torch.all(video_tensor[feature_num][frame_num] == 0):
                    print("ALL ZEROES FOUND")
                    print("Video: " + str(video_number) + " Image: " + str(frame_num) + " Feature: " + str(feature_num))
                    # Fill in item if all zeroes detected
                    model = FrameSequenceLSTM(input_dim, hidden_dim, output_dim, num_layers)
                    model.load_state_dict(torch.load(f"feature_{feature_num}_final.pth", map_location=device))
                    model.to(device)
                    model.eval()  # Set the model to evaluation mode
                    outputted_sequence = model(video_tensor[feature_num].unsqueeze(0)) # Requires (1, X, 32, 32), so i Unsqueeze, output is same size
                    #.cpu() and .detach() methods required for converting tensor to np array after gpu processing
                    frameID_to_latent_encodings[(video_number, frame_num)][feature_num, :, :] = outputted_sequence[0][frame_num].cpu().detach()
                    

def decode_full_frame_and_save_all(model, output_dir, device):
    """Decode the full frame from stored latent encodings and save for all video-image frames."""
    print(f"Total size of frameID to latent encodings dict: {len(frameID_to_latent_encodings)}")
    
    for (video_number, image_number), latent_encodings in frameID_to_latent_encodings.items():
        if latent_encodings is None:
            print(f"No latent encodings found for video {video_number}, image {image_number}")
            continue

        # Convert latent encodings to PyTorch tensor
        latent_tensor = torch.tensor(latent_encodings, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 10, 32, 32)
        
        # Decode the full frame
        with torch.no_grad():
            decoded_img = model.decode(latent_tensor)
        
        # Save the decoded image
        decoded_img = decoded_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"video_{video_number}_img_{image_number}.png")
        plt.imsave(output_path, decoded_img)
        print(f"Saved decoded full frame to {output_path}")


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 32 * 32  # Flattened frame size (idk if this is 32x32 or 28x28)
    hidden_dim = 128 # Tuned hyperparameter for LSTM
    output_dim = 32 * 32 #(THIS AND INPUT_DIM NEED TO BE CHANGED BASED ON ARCHITECTURE)
    num_layers = 2

    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    model = load_model(args.model_path, device) # Load the model
    
    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)
    print(f"Listening on {args.host}:{args.port}...")
    
    conn, addr = server_socket.accept()
    print(f"Connection established with {addr}")
    
    try:
        decode_and_store(conn)
        # feature_filler(device, input_dim, hidden_dim, output_dim, num_layers) # comment this to toggle feature filling vs no feature filling
        decode_full_frame_and_save_all(model, output_dir="received_and_decoded_frames_not_filled/", device=device)
    finally:
        conn.close()
        server_socket.close()
