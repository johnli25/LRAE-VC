import os
import torch
import socket
import argparse
import struct
import matplotlib.pyplot as plt
from models import PNC_Autoencoder, PNC_Autoencoder_with_Classification, LRAE_VC_Autoencoder, Compact_LRAE_VC_Autoencoder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Receiver Decoder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained autoencoder model (.pth)")
    parser.add_argument("--host", type=str, required=True, help="Host IP address to bind the server")
    parser.add_argument("--port", type=int, required=True, help="Port number to bind the server")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save decoded images")
    return parser.parse_args()


def load_model(model_path):
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model


def decode_and_save(model, output_dir, conn):
    """Decode received encoded features and save decoded images."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    
    while True:
        try:
            # Receive the size of the incoming data
            size_data = conn.recv(4)
            if not size_data:
                break  # End of transmission
            data_size = struct.unpack("I", size_data)[0]
            
            # Receive metadata and feature bytes
            data = conn.recv(data_size)
            metadata_bytes = data[:12]  # First 12 bytes are metadata (3 unsigned integers)
            feature_bytes = data[12:]  # Remaining bytes are the feature
            
            # Deserialize metadata
            video_number, image_number, feature_number = struct.unpack("III", metadata_bytes)
            
            # Deserialize the feature
            feature = torch.tensor(torch.frombuffer(feature_bytes, dtype=torch.float32)).reshape(1, 1, 32, 32)
            
            # Decode the feature map
            with torch.no_grad():
                decoded_img = model.decode(feature)
            
            # Save the decoded image
            decoded_img = decoded_img.squeeze(0).permute(1, 2, 0).numpy()  # Convert to HWC format for saving
            output_path = os.path.join(output_dir, f"video_{video_number}_img_{image_number}_feature_{feature_number}.png")
            plt.imsave(output_path, decoded_img)
            print(f"Saved decoded image to {output_path}")
        except Exception as e:
            print(f"Error during decoding or saving: {e}")
            break


if __name__ == "__main__":
    args = parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)
    print(f"Listening on {args.host}:{args.port}...")
    
    conn, addr = server_socket.accept()
    print(f"Connection established with {addr}")
    
    try:
        decode_and_save(model, args.output_dir, conn)
    finally:
        conn.close()
        server_socket.close()
