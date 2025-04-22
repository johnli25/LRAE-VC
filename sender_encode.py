import argparse, os, socket, struct, zlib
import cv2, torch, numpy as np
import torch.nn as nn
from models import PNC32Encoder, PNC32, ConvLSTM_AE
import json
import time

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
            model = PNC32Encoder()
        elif model_name == "conv_lstm_PNC32_ae":
            if lstm_kwargs is None:
                raise ValueError("Must pass --lstm_kwargs for ConvLSTM_AE state_dict")
            model = ConvLSTM_AE(**lstm_kwargs)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError(f"Unrecognized checkpoint type: {type(checkpoint)}")
    return model.to(device).eval()


def preprocess_frame(frame, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0 # normalize to [0, 1]

    t = torch.from_numpy(img) # convert to tensor
    t = t.permute(2, 0, 1) # NOTE: rearrange dimensions from (H, W, C) to (C, H, W)
    t = t.unsqueeze(0) # add batch dimension --> (1, C, H, W)
    t = t.to(device)
    return t

def encode_and_send(net, sock, frame, device, quantize, address, video_name):
    """
    Encodes and compresses one frame into latent features, optionally quantizes
    then sends with a 4-byte big-endian length prefix.
    """
    x = preprocess_frame(frame, device)
    with torch.no_grad():
        if hasattr(net, "encoder"): z = net.encoder(x) # conv_lstm_ae.encoder(x)
        else: z = net(x) # PNC32Encoder

        arr = z.cpu().numpy().astype(np.float32) # convert to numpy array

    # optional 8-bit quantization
    if quantize:
        arr = np.clip((arr * 255).round(), 0, 255).astype(np.uint8)
    
    payload = arr.tobytes()
    compressed_payload = zlib.compress(payload)
    packet = struct.pack("!I", len(compressed_payload)) + compressed_payload

    # Version 1: send whole frame (features) at once
    start_time = time.time()
    sock.sendto(packet, address)
    end_time = time.time()
    print(f"[sender] Sent packet of size {len(packet)} bytes in {end_time - start_time} seconds")

    # Version 2: send each feature separately
    total_features = arr.shape[1]  # arr shape is (1, 32, 32, 32) for PNC32
    print("Total features:", arr.shape) 
    start_time = time.time()
    for i in range(total_features):
        feature = arr[0, i]  # Extract the i-th feature
        payload = feature.tobytes()

        compressed_payload = zlib.compress(payload)
        print(f"[sender] Compressed feature {i} size: {len(compressed_payload)} bytes")

        packet = struct.pack("!I", len(compressed_payload)) + compressed_payload

        sock.sendto(packet, address)
    end_time = time.time()
    print(f"[sender] Version 2 by feature: Sent packet of size {len(packet)} bytes in {end_time - start_time} seconds")

    # NOTE: return purely for bookkeeping purposes: 
    return {
        "video_name": video_name,
        "frame": frame,
        "features": arr,
        "compressed_size": len(compressed_payload),
    }


def iter_videos(input_path):
    """Iterate over all video files in the input directory or a single video file."""
    if os.path.isdir(input_path):
        for fn in sorted(os.listdir(input_path)):
            if fn.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                cap = cv2.VideoCapture(os.path.join(input_path, fn))
                yield fn, cap
    else:
        yield os.path.basename(input_path), cv2.VideoCapture(input_path)


def main():
    parser = argparse.ArgumentParser(
        description="Encode frames with PNC32Encoder or ConvLSTM_AE and send."
    )
    parser.add_argument("--model",       required=True, choices=["pnc32","conv_lstm_PNC32_ae"])
    parser.add_argument("--model_path",  required=True,
                        help=".pth file (full model or state_dict)")
    parser.add_argument("--lstm_kwargs", type=str, default=None,
                        help="JSON dict for ConvLSTM_AE ctor if loading state_dict")
    parser.add_argument("--input",       required=True,
                        help="video file or directory of videos")
    parser.add_argument("--ip",          required=True, help="receiver IP")
    parser.add_argument("--port",        type=int, required=True, help="receiver port")
    # parser.add_argument("--quant", type=bool, default=False, help="enable or disable 8-bit quantization (default: False)")
    parser.add_argument("--quant", action="store_true", help="enable 8-bit quantization (default: False)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lstm_kwargs = json.loads(args.lstm_kwargs) if args.lstm_kwargs else None
    net = load_model(args.model, args.model_path, device, lstm_kwargs)

    # Create a UDP socket (no need to connect)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.ip, args.port)
    print(f"[sender] UDP socket ready to send to {dest}")

    test_videos = {
        "Diving-Side001", 
        # "Golf-Swing-Front_005", "Kicking-Front_003", # just use these video(s) for temporary results
        # "Lifting_002", "Riding-Horse_006", "Run-Side_001",
        # "SkateBoarding-Front_003", "Swing-Bench_016", "Swing-SideAngle_006", "Walk-Front_021"
    }
    try:
        for video_name, cap in iter_videos(args.input):
                if video_name.split(".")[0] not in test_videos:
                    # print(f"[sender] Skipping video: {video_name}")
                    continue
                print(f"[sender] Processing video: {video_name}")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    info = encode_and_send(net, sock, frame, device, args.quant, dest, video_name)
                    # print(f"[sender] Sent frame {info['video_name']} with compressed size {info['compressed_size']} bytes")
                cap.release()
    finally:
        sock.close()
        print("[sender] Done. Socket closed.")

if __name__ == "__main__":
    main()