import argparse, os, socket, struct, zlib
import cv2, torch, numpy as np
import torch.nn as nn
from models import PNC32Encoder, PNC32, ConvLSTM_AE
import json
import time
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict

def load_model(model_name, model_path, device, lstm_kwargs=None):
    """
    model_name: "pnc32" or "conv_lstm_PNC32_ae"
    model_path: .pth file, either a full-saved model or a state_dict
    lstm_kwargs: dict with keys (total_channels, hidden_channels, ae_model_name, bidirectional)
                 required only when model_name=="conv_lstm_PNC32_ae" and you saved state_dict
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, nn.Module): # If you saved the full model object:
        print("LOADING FULL MODEL OBJECT")
        model = checkpoint

    elif isinstance(checkpoint, dict): # Otherwise assume it's a state_dict
        if any(k.startswith("module.") for k in checkpoint.keys()): # If the model was saved with DataParallel, remove "module." prefix
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

        if model_name == "pnc32":
            model = PNC32()
        elif model_name == "conv_lstm_PNC32_ae":
            if lstm_kwargs is None:
                raise ValueError("Must pass --lstm_kwargs for ConvLSTM_AE state_dict")
            model = ConvLSTM_AE(**lstm_kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        # Filter out unexpected keys
        model_state_dict = model.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        missing_keys = [k for k in model_state_dict.keys() if k not in checkpoint]
        unexpected_keys = [k for k in checkpoint.keys() if k not in model_state_dict]

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Ignoring unexpected keys in checkpoint: {unexpected_keys}")
        
        model.load_state_dict(filtered_checkpoint)
    else:
        raise RuntimeError(f"Unrecognized checkpoint type: {type(checkpoint)}")
    return model.to(device).eval()


def save_img(rgb_tensor, outdir, idx):
    squeezed_tensor = rgb_tensor.squeeze(0) # Remove the batch dimension (squeeze the first dimension)
    permuted_tensor = squeezed_tensor.permute(1, 2, 0) # Rearrange the tensor dimensions from (C, H, W) to (H, W, C) for image representation
    img_array = permuted_tensor.cpu().numpy() # Move the tensor to the CPU and convert it to a NumPy array
    img_array = img_array.clip(0, 1) # Scale pixel values to [0, 1] 
    output_path = os.path.join(outdir, f"frame_{idx:04d}.png") 
    plt.imsave(output_path, img_array)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, choices=["pnc32","conv_lstm_PNC32_ae"])
    parser.add_argument("--model_path", required=True, help=".pth with PNC32Decoder weights")
    parser.add_argument("--port",        type=int, required=True)
    parser.add_argument("--ip",          default="0.0.0.0")
    parser.add_argument("--lstm_kwargs", type=str, default=None,
                        help="JSON dict for ConvLSTM_AE constructor if loading state_dict")
    parser.add_argument("--deadline_ms", type=float, default=30,
                    help="Deadline in milliseconds per frame")
    parser.add_argument("--quant", action="store_true", help="enable integer-bit quantization (default: False)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lstm_kwargs = json.loads(args.lstm_kwargs) if args.lstm_kwargs else None
    net = load_model(args.model, args.model_path, device, lstm_kwargs)

    output_dir = "./receiver_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 2**21)  # 2MB receive buffer
    recv_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Receive buffer size: {recv_buf} bytes")

    sock.bind((args.ip, args.port))
    sock.settimeout(1.0) # timeout of 1 second

    # Flush any leftover packets in the socket buffer
    # sock.setblocking(False)
    # try:
    #     while True:
    #         pkt, _ = sock.recvfrom(4096)
    #         print("Flushed leftover packet")
    # except BlockingIOError:
    #     # No more packets left to flush
    #     print("Probably finished.")
    # sock.setblocking(True)

    print(f"[receiver] Listening on {args.ip}:{args.port}")

    ##### Version 2: receive each feature separately #####
    features_dict = defaultdict(lambda: [np.zeros((32, 32)) for _ in range(32)])    
    frame_timestamp = None
    deadline_ms = args.deadline_ms  
    cur_frame = 0
    TOTAL_FEATURES = 32
    current_features_received = 0

    while True:
        # print("in while True: Waiting for packet...")
        try:
            pkt, _ = sock.recvfrom(4096)
            frame_idx, feature_num, data_len = struct.unpack_from("!III", pkt, 0)
            if frame_timestamp is None:
                frame_timestamp = time.monotonic_ns() / 1e6 # convert to milliseconds
            if cur_frame != frame_idx:  # IMPORTANT NOTE: if frame_idx is not the expected one, skip. This also drops frames that are 1) out of order or 2) arrived past the deadline! 
                continue

            data = pkt[12 : 12 + data_len]  # Extract the payload starting after the header
            # print(f"Received pkt for frame_idx: {frame_idx}, feature_num: {feature_num}, data_len: {data_len}")
            feature = zlib.decompress(data)
            feature = np.frombuffer(feature, dtype=np.uint8 if args.quant else np.float32)
            if args.quant:
                feature = feature.astype(np.float32) / 255.0
            feature = feature.reshape(32, 32) # Reshape to 
            features_dict[frame_idx][feature_num] = feature # NOTE: this is a list of 32 features, each 32x32
            current_features_received += 1
            # print(len(features_dict[frame_idx]), "features received")

            now = time.monotonic_ns() / 1e6 # convert to milliseconds
            if current_features_received == TOTAL_FEATURES or (now - frame_timestamp > args.deadline_ms): # NOTE: args.deadline_ms does NOT include the time to decode + display!
                print("now - frame_timestamp:", now - frame_timestamp)
                start_decode_time = time.monotonic() * 1000
                print(f"[receiver] Frame {frame_idx} received {current_features_received} features")
                latent = np.stack(features_dict[frame_idx])[None, ...] # NOTE: stack 32 feature maps (each 32x32) into a (32, 32, 32) array and [None, ...] adds batch dimension --> (1, 32, 32, 32)
                z = torch.from_numpy(latent).to(device).float() # Convert to float PyTorch tensor and move to device

                with torch.no_grad():
                    if hasattr(net, "decoder"):
                        z_seq = z.unsqueeze(1)  # (1, 1, 32, 32, 32)
                        lstm_out = net.conv_lstm(z_seq)

                        if net.project_lstm_to_latent is not None:
                            print("Projecting LSTM output to latent space")
                            hc = 2 * net.hidden_channels if net.bidirectional else net.hidden_channels
                            lat = net.project_lstm_to_latent(lstm_out.view(-1, hc, 32, 32))
                            lat = lat.view(1, 1, net.total_channels, 32, 32)
                        else:
                            lat = lstm_out

                        recon = net.decoder(lat[:, 0])  # shape: (1, 3, 224, 224)
                    else: 
                        recon = net.decode(z)
                end_decode_time = time.monotonic() * 1000
                print("Time to decode:", end_decode_time - start_decode_time)

                save_img(recon, output_dir, frame_idx)
                cur_frame = frame_idx + 1
                frame_timestamp = None
                current_features_received = 0


        except socket.timeout:
            if frame_timestamp and current_features_received > 0:
                print(f"[receiver:timeout] Frame {cur_frame} timed out with {current_features_received} features")
                features = features_dict[cur_frame]

                # Stack features into latent tensor
                latent = np.stack(features)[None, ...]  # shape: (1, 32, 32, 32)
                z = torch.from_numpy(latent).to(device).float()

                with torch.no_grad():
                    if hasattr(net, "decoder"):
                        z_seq = z.unsqueeze(1)  # (1, 1, 32, 32, 32)
                        lstm_out = net.conv_lstm(z_seq)

                        if net.project_lstm_to_latent is not None:
                            print("Projecting LSTM output to latent space")
                            hc = 2 * net.hidden_channels if net.bidirectional else net.hidden_channels
                            lat = net.project_lstm_to_latent(lstm_out.view(-1, hc, 32, 32))
                            lat = lat.view(1, 1, net.total_channels, 32, 32)
                        else:
                            lat = lstm_out

                        recon = net.decoder(lat[:, 0])
                    else:
                        recon = net.decode(z)

                save_img(recon, output_dir, cur_frame)
                cur_frame += 1
                frame_timestamp = None
                current_features_received = 0


if __name__ == "__main__":
    main()