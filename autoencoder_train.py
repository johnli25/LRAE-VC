import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from models import PNC16, PNC32
import random
from pytorch_msssim import ssim
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from tqdm import tqdm
from collections import OrderedDict
import csv


class PNC32WithEntropy(nn.Module):
    def __init__(self, pretrained=None, freeze_base=True):
        super().__init__()
        self.base = PNC32() # original ae
        if pretrained: 
            self.base.load_state_dict(torch.load(pretrained), strict=False)

        C = self.base.encoder2.out_channels  # 32
        # 1×1 convs for hyperprior (no need to modify PNC32)
        self.hyper_encoder = nn.Conv2d(C, C, kernel_size=1)
        self.hyper_decoder = nn.Conv2d(C, C, kernel_size=1)

        self.entropy_z = EntropyBottleneck(C)
        self.gauss_y = GaussianConditional(None)

        if freeze_base:
            for p in self.base.parameters(): p.requires_grad = False

    def forward(self, x, tail_length=None, quantize_level=0):
        y   = self.base.encode(x) # y is the original compressed/encoded latent feature 
        z   = self.hyper_encoder(y) # z captures additional information (e.g., statistics like variance) about 'y' to improve compression.
        z_q, z_lh = self.entropy_z(z) # z_q is the quantized version of z, and z_lh is the likelihood of z given the model.
        sigma = self.hyper_decoder(z_q) # sigma is used to model the distribution of 'y' (e.g., as a Gaussian with mean and variance).
        y_q, y_lh  = self.gauss_y(y, sigma) # y_q is the quantized version of y, and y_lh is the likelihood of y given the model, which will be fed into decoder/reconstructer
        recon  = self.base.decode(y_q)
        return recon, y_lh, z_lh 
    
    def compress(self, x):
        y = self.base.encode(x)
        z = self.hyper_encoder(y)
        z_bytes = self.entropy_z.compress(z)
        z_q = self.entropy_z.decompress(z_bytes)
        sigma = self.hyper_decoder(z_q)
        y_bytes = self.gauss_y.compress(y, sigma)
        return {"z": z_bytes, "y": y_bytes}

    def decompress(self, streams):
        z_q  = self.entropy_z.decompress(streams["z"])
        sigma = self.hyper_decoder(z_q)
        y_q  = self.gauss_y.decompress(streams["y"], sigma)
        return self.base.decode(y_q)


# NOTE: uncomment below if you're using UCF Sports Action 
class_map = {
    "diving": 0, "golf_front": 1, "kick_front": 2, "lifting": 3, "riding_horse": 4,
    "running": 5, "skating": 6, "swing_bench": 7
}

test_img_names = {
    "Diving-Side_001",
    "Golf-Swing-Front_005", 
    "Kicking-Front_003", 
    # "Lifting_002", "Riding-Horse_006", "Run-Side_001",
    # "SkateBoarding-Front_003", "Swing-Bench_016", "Swing-SideAngle_006", "Walk-Front_021"
}


# Dataset class for loading images and ground truths
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Use the same image for both input and ground truthx
        return image, self.img_names[idx]  # (image, same_image_as_ground_truth, img filename)


def train_autoencoder(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_name, max_tail_length, quantize=0, entropy=False):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    if max_tail_length: 
        print(f"Training with max tail length: {max_tail_length}")
        drops = -1

    for epoch in range(num_epochs):
        # increase the tail length
        if max_tail_length and epoch % 2 == 0:
            drops = min(drops + 1, max_tail_length)
            print(f"Epoch {epoch}: Increasing tail length to {drops}")

        # Train the model
        model.train()
        train_loss = 0
        for inputs, _ in train_loader:
            # Sample a single tail length for the batch
            # torch.manual_seed(seed=42)
            tail_len = random.randint(0, drops) if max_tail_length else None
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(x=inputs, tail_length=tail_len, quantize_level=quantize)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        val_loss, _, _ = eval_autoencoder(model, val_loader, criterion, device, max_tail_length, quantize)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if max_tail_length: 
                # only save w_taildrops if you actually used them
                fname = f"{model_name}_best_validation_w_taildrops.pth"
                torch.save(model.state_dict(), fname)
                print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved as {fname}")
            else:
                fname = f"{model_name}_best_validation_no_dropouts.pth"
                torch.save(model.state_dict(), fname)
                print(f"Epoch [{epoch+1}/{num_epochs}]: Validation loss improved. Model saved as {fname}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    plot_train_val_loss(train_losses, val_losses)

    # Save final model
    if max_tail_length:
        fname = f"{model_name}_final_w_taildrops.pth"
    else:
        fname = f"{model_name}_final_no_dropouts.pth"
    torch.save(model.state_dict(), fname)
    print(f"Final model saved as {fname}")

    # Final evaluation/test on test set
    test_loss, _, _ = eval_autoencoder(model, test_loader, criterion, device, max_tail_length, quantize)
    print(f"Final Test Loss: {test_loss:.4f}")


def train_entropy_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    img_height,
    img_width,
    lambda_rate=0.01,
    num_epochs=1,
    model_name="something_with_pth"
):
    """
    Trains the entropy-augmented model for num_epochs over train_loader.
    model        : an instance of PNC32WithEntropy
    train_loader : DataLoader yielding (x, _) batches
    criterion    : e.g. nn.MSELoss()
    optimizer    : optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), ...)
    device       : torch.device
    img_height   : H (e.g. 224)
    img_width    : W (e.g. 224)
    lambda_rate  : trade-off weight for rate term
    num_epochs   : how many full passes over the data
    """
    model_name = model_name.replace(".pth", "")  # Remove .pth if present
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):        
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            recon, y_lh, z_lh = model(x)

            # distortion term
            mse = criterion(recon, x)

            # rate term (bits per pixel)
            bits_y = (-torch.log2(y_lh + 1e-9)).sum()
            bits_z = (-torch.log2(z_lh + 1e-9)).sum()
            bpp = (bits_y + bits_z) / (x.size(0) * img_height * img_width)

            loss = mse + lambda_rate * bpp
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] avg loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"{model_name}_entropy_final.pth")
    return model


def eval_autoencoder(model, dataloader, criterion, device, max_tail_length=None, quantize=0):
    model.eval()
    total_mse = 0.0
    total_ssim = 0.0
    num_samples = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            # torch.manual_seed(seed=42)
            # tail_len = torch.randint(0, max_tail_length, (1,)).item() if max_tail_length else None
            # print("Eval tail length: ", tail_len)
            inputs = inputs.to(device)

            if not args.entropy: # AKA if PNC32
                outputs = model(x=inputs, tail_length=max_tail_length, quantize_level=quantize)
            elif args.entropy: # AKA if PNC32WithEntropy
                outputs, _, _ = model(x=inputs, tail_length=max_tail_length, quantize_level=quantize)

            ##### NOTE: "intermission" function: print estimated byte size of compressed latent features
            # frame_latent = model.module.encode(inputs[0])
            # if quantize > 0:    
            #     features_cpu = frame_latent.detach().cpu().numpy()
            #     features_uint8 = (features_cpu * 255).astype(np.uint8)  # Convert to uint8
            #     compressed = zlib.compress(features_uint8.tobytes())
            #     latent_num_bytes = len(compressed)
            #     print(f"[Simulated Compression] Frame 0 compressed size (quantized to uint8): {latent_num_bytes} bytes "
            #         f"(Original shape: {tuple(frame_latent.shape)})")
            # else:
            #     features_cpu = frame_latent.detach().cpu().numpy().astype(np.float32)
            #     compressed = zlib.compress(features_cpu.tobytes())
            #     latent_num_bytes = len(compressed)
                # print(f"[Simulated Compression] Frame 0 compressed size (float32): {latent_num_bytes} bytes "
                #     f"(Original shape: {tuple(frame_latent.shape)})")
            ##############

            # Compute MSE loss
            mse_loss = criterion(outputs, inputs)
            total_mse += mse_loss.item() * inputs.size(0)

            # Compute SSIM
            ssim_value = ssim(inputs, outputs, data_range=1.0, size_average=False)
            total_ssim += ssim_value.sum().item()

            num_samples += inputs.size(0)

    assert len(dataloader.dataset) == num_samples, "Mismatch error between dataset size and number of samples processed"
    average_mse = total_mse / num_samples
    average_psnr = - 10 * np.log10(average_mse)
    average_ssim = total_ssim / num_samples

    return average_mse, average_psnr, average_ssim


def plot_train_val_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')
    
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig('train_val_loss_curve.png', dpi=300)
    plt.show()

def psnr(mse):
    return 20 * np.log10(1) - 10 * np.log10(mse)


def smart_load(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    entropy: bool = False,
    use_dataparallel: bool = False,) -> torch.nn.Module:

    ckpt = torch.load(checkpoint_path, map_location=device)

    # If you saved a dict with 'state_dict' key; else load whole checkpoint
    state_dict = ckpt.get('state_dict', ckpt)

    # 2. Build new state dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # strip any DataParallel prefix (we can add it later)
        k_stripped = k.replace('module.', '') 
        # if we're loading into PNC32WithEntropy and it's a base-AE key, prefix with 'base.'
        if entropy and not ( # NOTE: for PNC32Entropy, .base needs to be prepended first! 
            k_stripped.startswith('hyper_encoder') or
            k_stripped.startswith('hyper_decoder') or
            k_stripped.startswith('entropy_z') or
            k_stripped.startswith('gauss_y')
        ):
            new_key = 'base.' + k_stripped # 
        else:
            new_key = k_stripped
        new_state_dict[new_key] = v

    # 3. Load into model allowing missing keys for new entropy layers
    model.load_state_dict(new_state_dict, strict=False)

    # 4. Wrap in DataParallel if desired
    if use_dataparallel and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)  # :contentReference[oaicite:1]{index=1}

    return model


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Train the PNC Autoencoder or PNC Autoencoder with Classification.")
        parser.add_argument("--model", type=str, required=True, choices=["PNC", "PNC_256U", "PNC16", "PNC32", "TestNew", "TestNew2", "TestNew3", "PNC_NoTail", "PNC_with_classification", "LRAE_VC"], 
                            help="Model to train")
        parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
        parser.add_argument("--epochs", type=int, default=28, help="Number of epochs to train")
        parser.add_argument("--quantize", type=int, default=0, help="Quantize latent features by how many bits/levels")
        parser.add_argument("--drops", type=int, default=0, help="Maximum tail length for feature random dropout")
        parser.add_argument("--entropy", action="store_true", help="Use entropy coding")
        return parser.parse_args()
    
    args = parse_args()

    ## Hyperparameters
    drops = args.drops
    num_epochs = args.epochs
    batch_size = 32
    learning_rate = 1e-3
    img_height, img_width = 224, 224 # NOTE: Dependent on autoencoder architecture!!
    path = "TUCF_sports_action_224x224/" # NOTE: already resized to 224x224 (so not really adaptable), but faster
    # path = "UCF_224x224x3_PNC_FrameCorr_input_imgs/" 
    # path = "UCF_uncompressed_video_img_frames" # NOTE: more adaptable to different img dimensions because it's the original, BUT SLOWER!

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)), # NOTE: not needed if dataset already resized (e.g. to 224x224)
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(path, transform=transform)

    test_indices = [
        i for i in range(len(dataset))
        if "_".join(dataset.img_names[i].split("_")[:-1]) in test_img_names
    ]
    train_val_indices = [i for i in range(len(dataset)) if i not in test_indices]

    # Split train_val_indices into train and validation
    np.random.shuffle(train_val_indices)
    train_size = int(0.9 * len(train_val_indices))
    val_size = len(train_val_indices) - train_size

    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # apply data augmentation to the train data
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform        
    test_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "PNC":
        # model = PNC_Autoencoder()
        max_tail_length = 10 # true PNC

    if args.model == "PNC16":
        model = PNC16()

    if args.model == "PNC32":
        if args.entropy:
            model = PNC32WithEntropy(pretrained = args.model_path)
        else:
            model = PNC32()

    model = model.to(device)

    use_dataparallel = torch.cuda.device_count() > 1
    print(f"Using DataParallel: {use_dataparallel} with {torch.cuda.device_count()} GPUs")
    # if args.model_path exists, load and continue training or evaluate from there
    if args.model_path:
        model = smart_load(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
        entropy=args.entropy,
        use_dataparallel=use_dataparallel,)    

    criterion = nn.MSELoss()

    # train the model
    if args.entropy: # if trained with Entropy
        # Stage 1: entropy-only
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )
        model = train_entropy_model(
            model, train_loader, criterion, optimizer,
            device, img_height, img_width, lambda_rate=0.000001, num_epochs=7, model_name=args.model_path
        )
        # Stage 2: fine-tune
        if use_dataparallel:    
            for p in model.module.base.parameters():
                p.requires_grad = True
        else:
            for p in model.base.parameters(): 
                p.requires_grad = True
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )   
        model = train_entropy_model(
            model, train_loader, criterion, optimizer,
            device, img_height, img_width, lambda_rate=0.000001, num_epochs=40, model_name=args.model_path
        )
    else: # plain PNC training (NO ENTROPY)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_autoencoder(
            model, train_loader, val_loader, test_loader,
            criterion, optimizer, device, num_epochs,
            args.model, max_tail_length=drops, quantize=args.quantize
        )


    # NOTE: uncomment below for hardcoded tail_len_drops for more AUTOMATED evaluation. Otherwise, leave commented in!
    # tail_len_drops = [0, 3, 6, 10, 13, 16, 19, 22, 26, 28]
    # mse_list, psnr_list, ssim_list = [], [], []
    # for tail_len_drop in tail_len_drops:
    #     print(f"Tail length: {tail_len_drop}")
    #     final_test_loss, final_psnr, final_ssim = eval_autoencoder(model=model, dataloader=test_loader, criterion=criterion, device=device, max_tail_length=tail_len_drop, quantize=args.quantize)
    #     mse_list.append(final_test_loss)
    #     psnr_list.append(final_psnr)
    #     ssim_list.append(final_ssim)
    #     print(f"Final Test Loss: {final_test_loss:.6f} and PSNR: {final_psnr:.6f} and SSIM: {final_ssim:.6f}")

    # csv_file = "PNC_results.csv"

    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['tail_len_drop', 'MSE', 'PSNR', 'SSIM'])  # header
    #     for i in range(len(tail_len_drops)):
    #         writer.writerow([tail_len_drops[i], mse_list[i], psnr_list[i], ssim_list[i]])

    final_test_loss, final_psnr, final_ssim = eval_autoencoder(model=model, dataloader=test_loader, criterion=criterion, device=device, max_tail_length=drops, quantize=args.quantize)
    print(f"Final Test Loss: {final_test_loss:.6f} and PSNR: {final_psnr:.6f} and SSIM: {final_ssim:.6f}")

    # Save images generated by DECODER ONLY! 
    output_path = "PNC_test_imgs_post_training/"
    if not os.path.exists(output_path):
        print(f"Creating directory: {output_path}")
        os.makedirs(output_path)

    model.eval()  # Put the autoencoder in eval mode
    with torch.no_grad():
        for i, (inputs, filenames) in enumerate(test_loader):
            inputs = inputs.to(device) 
            if args.entropy: # PNC32WithEntropy 
                outputs, y_lh, z_lh = model(inputs, tail_length=0, quantize_level=args.quantize)  # Forward pass through autoencoder
            else: # PNC32
                outputs = model(inputs, tail_length=0, quantize_level=args.quantize) # NOTE: tail_length basically means drop tail! ,

            # outputs is (batch_size, 3, image_h, image_w)
            print(f"Batch {i+1}/{len(test_loader)}, Output shape: {outputs.shape}")
            # Save each reconstructed image
            for j in range(inputs.size(0)):
                output_np = outputs[j].permute(1, 2, 0).cpu().numpy()  # (image_h, image_w, 3)
                plt.imsave(os.path.join(output_path, filenames[j]), output_np)
