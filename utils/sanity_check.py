import os
import numpy as np
import torch

video_features_dir = "./PNC16_combined_features" # if running in LRAE-VC (one directory above)
# video_features_dir = "../PNC16_combined_features" # if running in utils (where sanity_check.py is stored)
video_features_lookup = {}

# Check for ALL zero feature vectors
for filename in os.listdir(video_features_dir):
    video_tensor =  np.load(os.path.join(video_features_dir, filename))
    # print(f"{filename} has shape {video_tensor.shape}")
    for frame in video_tensor:
        for feature in frame:
            if np.all(feature == 0):
                print(f"{filename} has a zero feature vector of shape {feature.shape}")
                break
