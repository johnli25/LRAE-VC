#!/usr/bin/env python
import os
import cv2
import argparse
import re

def natural_sort_key(s):
    """
    A key function for natural sorting.
    Splits the string into a list of integers and non-numeric parts.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def group_images_by_prefix(input_dir):
    """
    Groups image filenames in input_dir by the prefix (first two parts split by '_').
    Returns a dictionary mapping prefix -> list of filenames.
    """
    groups = {}
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Assuming the filename has at least two parts separated by '_'
            parts = file.split('_')
            if len(parts) >= 2:
                prefix = parts[0] + parts[1]
                groups.setdefault(prefix, []).append(file)
    return groups

def create_video_for_group(input_dir, group_name, file_list, output_dir, fps=30):
    """
    Creates an MP4 video from the images in file_list (sorted naturally) for a given group.
    Assumes all images have the same resolution.
    """
    # Use natural sorting to correctly order filenames with numbers.
    file_list = sorted(file_list, key=natural_sort_key)
    
    # Read the first image to determine frame size
    first_image_path = os.path.join(input_dir, file_list[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading image {first_image_path}")
        return
    height, width, channels = frame.shape

    # Define the codec and create a VideoWriter object
    output_path = os.path.join(output_dir, f"{group_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for filename in file_list:
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read {image_path}")
            continue
        # Resize if necessary (optional)
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        out.write(img)
    out.release()
    print(f"Created video: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of images into MP4 videos grouped by filename prefix."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="output_videos", help="Directory to save output videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output videos")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    groups = group_images_by_prefix(args.input_dir)
    
    for prefix, files in groups.items():
        print(f"Processing group: {prefix} with {len(files)} images")
        create_video_for_group(args.input_dir, prefix, files, args.output_dir, fps=args.fps)

if __name__ == "__main__":
    main()
