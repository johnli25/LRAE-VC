import argparse, os, socket, struct, zlib
import cv2, torch, numpy as np
import torch.nn as nn
from models import PNC32Encoder, PNC32, ConvLSTM_AE
import json
import time
import matplotlib.pyplot as plt
import shutil

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
        if any(k.startswith("module.") for k in checkpoint.keys()):
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
    parser.add_argument("--deadline_ms", type=int, default=30,
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
    sock.bind((args.ip, args.port))
    sock.settimeout(0.01) # timeout is 0.01 sec or 10 ms
    print(f"[receiver] Listening on {args.ip}:{args.port}")

    deadline_sec = args.deadline_ms / 1000.0 # convert b/c Python's time is in seconds

    ##### Version 2: receive each feature separately #####
    features = []
    frame_timestamp = None
    frame_idx = 0
    while True:
        try:
            pkt, _ = sock.recvfrom(8192)
            print("length of pkt", len(pkt))
            if frame_timestamp is None:
                # unpack an unsigned 64‐bit timestamp from bytes 0–7
                t0_ns = struct.unpack_from("!Q", pkt, 0)[0]
                # frame_timestamp = t0_ns / 1000000.0   # convert to ms
                frame_timestamp = time.monotonic_ns() / 1000000.0 
                offset = 8  # skip the full 8‐byte header
            else:
                offset = 0
            now = time.monotonic_ns() / 1000000.0 # convert to ms
            
            data_len = struct.unpack_from("!I", pkt, offset)[0]
            print("data_len:", data_len)
            data = pkt[offset + 4 : offset + 4 + data_len] # 4 bytes for length of data packet
            feature = data # zlib.decompress(data)
            feature = np.frombuffer(feature, dtype=np.uint8 if args.quant else np.float32)
            print("feature shape:", feature.shape)
            if args.quant:
                feature = feature.astype(np.float32) / 255.0
            features.append(feature.reshape(32, 32))
            print(len(features), "features received")
            print("now - frame_timestamp:", now - frame_timestamp)
            print("now: ", now)
            print("frame_timestamp: ", frame_timestamp)
            if len(features) == 32 or (now - frame_timestamp > args.deadline_ms): # NOTE: args.deadline_ms does NOT include the time to decode + display!
                if len(features) < 32:
                    print("not enough features, padding with zeros")
                    features += [np.zeros((32, 32), dtype=np.float32)] * (32 - len(features))
                latent = np.stack(features)[None, ...] # NOTE: stack 32 feature maps (each 32x32) into a (32, 32, 32) array and [None, ...] adds batch dimension --> (1, 32, 32, 32)
                z = torch.from_numpy(latent).to(device) # convert to tensor and move to GPU

                with torch.no_grad():
                    recon = net.decoder(z) if hasattr(net, "decoder") else net.decode(z)

                save_img(recon, output_dir, frame_idx)
                print(f"[receiver:v2] Frame {frame_idx} received {len(features)} features")
                frame_idx += 1
                features.clear()
                frame_timestamp = None

        except socket.timeout:
            if frame_timestamp and features:
                if len(features) < 32:
                    features += [np.zeros((32, 32), dtype=np.float32)] * (32 - len(features))
                latent = np.stack(features)[None, ...]
                z = torch.from_numpy(latent).to(device)

                with torch.no_grad():
                    recon = net.decoder(z) if hasattr(net, "decoder") else net.decode(z)

                save_img(recon, output_dir, frame_idx)
                print(f"[receiver:v2] Frame {frame_idx} (timeout with {len(features)} features)")
                frame_idx += 1
                features.clear()
                frame_timestamp = None

if __name__ == "__main__":
    main()

##### Deprecated ###### 
@deprecated
def decode_super_pkt(payload, quant=False):
    """Version 3"""
    N = struct.unpack_from("!I", payload, 0)[0] # N = number of features
    offsets = struct.unpack_from(f"!{N}I", payload, 4)
    data_start = 4 + 4 * N
    print("offsets:", offsets)
    print("data_start:", data_start)

    features = []
    for i in range(N):
        start = data_start + offsets[i]
        end = data_start + offsets[i+1] if i < N-1 else None
        feature = payload[start:end]
        feature = zlib.decompress(feature)
        feature = np.frombuffer(feature, dtype=np.uint8 if quant else np.float32) # if quantized, use uint8 initially and then convert to float32
        if quant:
            feature = feature.astype(np.float32) / 255.0
        features.append(feature.reshape(32, 32))

    latent = np.stack(features, axis=0) # stack 32 feature maps (each 32x32) into a (32, 32, 32) array
    return np.expand_dims(latent, axis=0) # add batch dimension --> (1, 32, 32, 32)