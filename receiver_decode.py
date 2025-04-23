import argparse, os, socket, struct, zlib
import cv2, torch, numpy as np
import torch.nn as nn
from models import PNC32Encoder, PNC32, ConvLSTM_AE
import json
import time
import matplotlib.pyplot as plt
import shutil

def load_model(kind, path, device, lstm_kwargs=None):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        # strip "module." for DataParallel checkpoints
        if any(k.startswith("module.") for k in ckpt):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

        if kind == "pnc32":
            model = PNC32Encoder()          # only encoder – fine; we won't reconstruct
        elif kind == "conv_lstm_PNC32_ae":
            if lstm_kwargs is None:
                raise ValueError("--lstm_kwargs missing")
            model = ConvLSTM_AE(**lstm_kwargs)
        else:
            raise ValueError("unknown model kind")
        model.load_state_dict(ckpt)

    return model.to(device).eval()


def save_img(rgb_tensor, outdir, idx):
    squeezed_tensor = rgb_tensor.squeeze(0) # Remove the batch dimension (squeeze the first dimension)
    permuted_tensor = squeezed_tensor.permute(1, 2, 0) # Rearrange the tensor dimensions from (C, H, W) to (H, W, C) for image representation
    img_array = permuted_tensor.cpu().numpy() # Move the tensor to the CPU and convert it to a NumPy array
    img_array = img_array.clip(0, 1) # Scale pixel values to [0, 1] 
    output_path = os.path.join(outdir, f"frame_{idx:04d}.png") 
    plt.imsave(output_path, img_array)
    
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

def decode_feature_packet(pkt, quant):
    """Version 2: Decode a single compressed 32x32 feature map."""
    pkt_len = struct.unpack_from("!I", pkt, 0)[0]
    raw = zlib.decompress(pkt[4:])
    arr = np.frombuffer(raw, dtype=np.uint8 if quant else np.float32)
    if quant:
        arr = arr.astype(np.float32) / 255.0
    return arr.reshape(32, 32)


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

    output_dir = "receiver_output"
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
            pkt, _ = sock.recvfrom(4096)
            print("len of pkt", len(pkt))
            now = time.time() * 1000.0 # in milliseconds

            if frame_timestamp is None:  # first packet of frame
                frame_timestamp  = struct.unpack_from("!I", pkt, 0)[0] # ms from sender
                offset = 4 # skip timestamp
            else:
                offset = 0 # no timestamp
            
            pkt_len = struct.unpack_from("!I", pkt, offset)[0]
            data = pkt[offset + 4 : offset + 4 + pkt_len] # 4 bytes for length of data packet
            feature = zlib.decompress(data)
            feature = np.frombuffer(feature, dtype=np.uint8 if args.quant else np.float32)
            if args.quant:
                feature = feature.astype(np.float32) / 255.0
            features.append(feature.reshape(32, 32))

            if len(features) == 32 or (now - frame_timestamp > args.deadline_ms): # NOTE: args.deadline_ms does NOT include the time to decode + display!
                if len(features) < 32:
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
