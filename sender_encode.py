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
        if hasattr(net, "encoder"): 
            z = net.encoder(x) # e.g. conv_lstm_PNC32_ae.encoder(x)
        else: 
            z = net.encode(x) # PNC32

        arr = z.cpu().numpy().astype(np.float32) # convert to numpy array

    # optional 8-bit quantization
    if quantize:
        arr = np.clip((arr * 255).round(), 0, 255).astype(np.uint8)

    # Version 2: send each feature separately
    total_features = arr.shape[1]  # NOTE: arr.shape is (1, 32, 32, 32) for PNC32

    capture_ts = int(time.time() * 1000.0) # capture_ts in milliseconds
    start_time = time.time()
    capture_ts = struct.pack("!I", capture_ts & 0xFFFFFFFF) # ensures 4-byte length

    first_feat = arr[0, 0].tobytes()
    first_body = zlib.compress(first_feat)
    first_pkt  = capture_ts + struct.pack("!I", len(first_body)) + first_body
    print(len(first_pkt), len(first_pkt))
    sock.sendto(first_pkt, address)

    for i in range(1, total_features):
        feature = arr[0, i]  # Extract the i-th feature

        compressed_payload = zlib.compress(feature.tobytes())
        # print(f"[sender] Compressed feature {i} size: {len(compressed_payload)} bytes")

        packet = struct.pack("!I", len(compressed_payload)) + compressed_payload
        # print(len(packet), len(compressed_payload)) 
        sock.sendto(packet, address)
    end_time = time.time()
    print(f"Version 2 [sender]: Sent {total_features} features in {end_time - start_time:.6f} seconds")

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
                        help="JSON dict for ConvLSTM_AE constructor if loading state_dict")
    parser.add_argument("--input",       required=True, help="video file or directory of videos")
    parser.add_argument("--ip",          required=True, help="receiver IP")
    parser.add_argument("--port",        type=int, required=True, help="receiver port")
    # parser.add_argument("--quant", type=bool, default=False, help="enable or disable 8-bit quantization (default: False)")
    parser.add_argument("--quant", action="store_true", help="enable integer-bit quantization (default: False)")

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



##### Deprecated stuff #####
# Version 1: Send whole frame (features) at once
if False:  # Deprecated code block
    # Version 1: Send whole frame (features) at once
    start_time = time.time()
    compressed_payload = zlib.compress(arr.tobytes())
    packet = struct.pack("!I", len(compressed_payload)) + compressed_payload

    sock.sendto(packet, address)
    end_time = time.time()
    print(f"Version 1 [sender]: Sent {len(arr)} features in {end_time - start_time:.6f} seconds")


# Version 3: send whole frame (features) but with offsets between features 
if False:
    offsets = []
    cursor = 0
    payload_chunks = []

    start_time = time.time()
    for i in range(total_features):
        feature = arr[0, i]
        compressed_payload = zlib.compress(feature.tobytes())
        # print(f"[sender] Compressed feature {i} size: {len(compressed_payload)} bytes")

        offsets.append(cursor)
        payload_chunks.append(compressed_payload)
        cursor += len(compressed_payload)

    header = struct.pack("!I", len(offsets)) # num features (4 B) e.g. = 32 features 
    header += b"".join(struct.pack("!I", offset) for offset in offsets) # e.g. 4 (total length) + 32 x 4 = 132 B
    body = b"".join(payload_chunks) 
    packet = struct.pack("!I", len(header) + len(body)) + header + body
    # print("packet is of length: ", len(packet))
    sock.sendto(packet, address)  

    end_time = time.time()
    print(f"[Version 3: sender] Sent {len(offsets)} features in {end_time - start_time:.6f} seconds")