import os
from PIL import Image

def preprocessDataset(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_path, filename)
            with Image.open(img_path) as img:
                img = img.resize((256, 256))  # Resize to 224x224
                img = img.convert("RGB")  # Ensure 3 channels (RGB)
                
                # Save the resized image directly as uint8 JPEG
                output_file_path = os.path.join(output_path, filename)
                img.save(output_file_path, format='JPEG')

# Usage
input_path = 'UCF_uncompressed_video_img_frames'
output_path = 'UCF_256x256x3_PNC_FrameCorr_input_imgs'
preprocessDataset(input_path, output_path)
