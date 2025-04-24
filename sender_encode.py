import argparse, os, socket, struct, zlib
import cv2, torch, numpy as np
import torch.nn as nn
from models import PNC32Encoder, PNC32, ConvLSTM_AE
import json
import time
import matplotlib.pyplot as plt

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
    # img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0 # normalize to [0, 1]
    # plt.imshow(img)
    # plt.title("Original Frame")
    # plt.axis("off")  # Optional: Turn off axis labels
    # plt.imsave("frame.png", img)  # Save the original frame as an image

    t = torch.from_numpy(img) # convert to tensor
    t = t.permute(2, 0, 1) # NOTE: rearrange dimensions from (H, W, C) to (C, H, W)
    t = t.unsqueeze(0) # add batch dimension --> (1, C, H, W)
    t = t.to(device)
    return t


def save_img(rgb_tensor, outdir, idx):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    squeezed_tensor = rgb_tensor.squeeze(0) # Remove the batch dimension (squeeze the first dimension)
    permuted_tensor = squeezed_tensor.permute(1, 2, 0) # Rearrange the tensor dimensions from (C, H, W) to (H, W, C) for image representation
    img_array = permuted_tensor.cpu().numpy() # Move the tensor to the CPU and convert it to a NumPy array
    img_array = img_array.clip(0, 1) # Scale pixel values to [0, 1] 
    output_path = os.path.join(outdir, f"frame_{idx:04d}.png") 
    plt.imsave(output_path, img_array)


def encode_and_send(net, sock, frame, device, quantize, address, video_name, frame_idx):
    """
    Encodes and compresses one frame into latent features, optionally quantizes
    then sends with a 4-byte big-endian length prefix.
    """
    start_time = time.monotonic_ns() / 1e6  
    x = preprocess_frame(frame, device)

    with torch.no_grad():
        if hasattr(net, "encoder"): 
            z = net.encoder(x) # e.g. conv_lstm_PNC32_ae.encoder(x)
            
            ## NOTE: optional sanity check! Save the original frame as an image
            # z_seq = z.unsqueeze(1)  # (1, 1, 32, 32, 32)
            # lstm_out = net.conv_lstm(z_seq)

            # if net.project_lstm_to_latent is not None:
            #     print("Projecting LSTM output to latent space")
            #     hc = 2 * net.hidden_channels if net.bidirectional else net.hidden_channels
            #     lat = net.project_lstm_to_latent(lstm_out.view(-1, hc, 32, 32))
            #     lat = lat.view(1, 1, net.total_channels, 32, 32)
            # else:
            #     lat = lstm_out

            # recon = net.decoder(lat[:, 0])  # shape: (1, 3, 224, 224)
            # save_img(recon, "sender_frames/", frame_idx)
            
        else: 
            z = net.encode(x) # PNC32

            ## NOTE: optional sanity check! Save the original frame as an image
            # recon = net.decode(z) # reconstruction
            # save_img(recon, "sender_frames/", frame_idx) # Save the original frame as an image

        arr = z.cpu().numpy().astype(np.float32) # convert to numpy array

    # optional 8-bit quantization
    if quantize:
        arr = np.clip((arr * 255).round(), 0, 255).astype(np.uint8)

    # Version 2: send each feature separately
    total_features = arr.shape[1]  # NOTE: arr.shape is (1, 32, 32, 32) for PNC32

    for i in range(total_features):
        feature = arr[0, i]  # Extract the i-th feature
        compressed_payload = zlib.compress(feature.tobytes())
        # print("frame_idx and feature:", frame_idx, i)
        packet = struct.pack("!III", frame_idx, i, len(compressed_payload)) + compressed_payload
        sock.sendto(packet, address)
        # time.sleep(0.001) # 1ms delay between packets

    end_time = time.monotonic_ns() / 1e6
    print(f"[sender] Encoded and Sent {total_features} features in {end_time - start_time:.6f} ms for frame {frame_idx} of video {video_name}")

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

    print("[sender] Running warm-up pass to help reduce latency significantly on first frame.")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        if hasattr(net, "encoder"):
            _ = net.encoder(dummy)
        else:
            _ = net.encode(dummy)
    print("[sender] Warm-up complete.")

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
        frame_idx = 0
        for video_name, cap in iter_videos(args.input):
            if video_name.split(".")[0] not in test_videos:
                # print(f"[sender] Skipping video: {video_name}")
                continue
            print(f"[sender] Processing video: {video_name}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _ = encode_and_send(net, sock, frame, device, args.quant, dest, video_name, frame_idx)
                frame_idx += 1
            cap.release()
    finally:
        sock.close()
        print("[sender] Done. Socket closed.")

if __name__ == "__main__":
    main()