import os
import cv2
import random
import shutil

def is_25fps(video_path, target_fps=25, tol=0.5):
    """
    Open the video and check if its FPS is approximately equal to target_fps.
    tol: tolerance to account for slight variations.
    Returns True if video FPS is within tol of target_fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open {video_path}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return abs(fps - target_fps) < tol

def create_subset_dataset(original_dir, new_dataset_dir, fraction=0.1, target_fps=25):
    """
    Traverse the original dataset directory, collect all video files (.avi or .mp4)
    that have approximately target_fps. Then randomly select the given fraction
    (e.g., 10%) of these videos, and copy them into new_dataset_dir.
    """
    valid_videos = []
    # Walk through the entire directory tree.
    for dirpath, dirnames, filenames in os.walk(original_dir):
        for filename in filenames:
            lower_filename = filename.lower()
            if lower_filename.endswith('.avi') or lower_filename.endswith('.mp4'):
                video_path = os.path.join(dirpath, filename)
                if is_25fps(video_path, target_fps=target_fps):
                    valid_videos.append(video_path)
    
    total_valid = len(valid_videos)
    print(f"Found {total_valid} videos with ~{target_fps} FPS.")
    
    # Calculate how many videos to select (at least 1 if valid_videos is not empty)
    num_to_select = max(1, int(total_valid * fraction)) if total_valid > 0 else 0
    selected_videos = random.sample(valid_videos, num_to_select)
    print(f"Selecting {num_to_select} videos (10% subset).")
    
    # Create the new dataset directory if it doesn't exist.
    os.makedirs(new_dataset_dir, exist_ok=True)
    
    # Copy each selected video to the new dataset directory.
    for video in selected_videos:
        # You may want to preserve the original filename or modify it;
        # here we just copy the file using its original filename.
        dest_path = os.path.join(new_dataset_dir, os.path.basename(video))
        print(f"Copying:\n  {video}\n  -> {dest_path}")
        shutil.copy2(video, dest_path)
    
    print("Subset dataset creation complete.")

if __name__ == "__main__":
    # Adjust these paths as necessary:
    original_dataset_dir = "./UCF101/UCF-101/"  # original dataset folder
    new_dataset_dir = "./New_UCF101_Videos"               # new folder for the subset
    create_subset_dataset(original_dataset_dir, new_dataset_dir, fraction=0.05, target_fps=25)
