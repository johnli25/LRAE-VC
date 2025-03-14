import cv2
import os
import re

def extract_32_frames_from_videos(
    video_dir, 
    output_dir, 
    num_frames=32,
    file_ext=".jpg"
):
    """
    Extract exactly 'num_frames' frames from each .mp4 video in 'video_dir'
    and save them to 'output_dir' with filename format: action_videoNum_frameNum.ext.
    Each frame is resized to 224×224 before saving.
    
    Videos with fewer than 'num_frames' frames are skipped.
    
    Args:
        video_dir (str): Path to directory containing .mp4 videos.
        output_dir (str): Path to directory where frames will be saved.
        num_frames (int): Number of frames to extract per video (default=32).
        file_ext (str): File extension for output images (default='.jpg').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Regex to match filenames like 'diving4.mp4' or 'golf_front7.mp4'
    video_pattern = re.compile(r"^(.+?)(\d+)\.mp4$")

    for filename in os.listdir(video_dir):
        if filename.lower().endswith(".mp4"):
            match = video_pattern.match(filename)
            if not match:
                print(f"Skipping '{filename}' because it doesn't match the pattern (action + number).")
                continue

            action = match.group(1)        # e.g., 'diving' or 'golf_front'
            video_num = match.group(2)       # e.g., '4'
            
            video_path = os.path.join(video_dir, filename)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Skip videos with fewer than num_frames frames
            if total_frames < num_frames:
                print(f"Skipping '{filename}' because it has only {total_frames} frames (less than {num_frames}).")
                cap.release()
                continue

            if total_frames <= 0:
                print(f"Warning: '{filename}' has zero frames or cannot be read.")
                cap.release()
                continue

            # Compute interval for uniform sampling
            interval = total_frames // num_frames

            print(f"Processing: {filename} (action='{action}', video_num='{video_num}', total_frames={total_frames})")

            for i in range(num_frames):
                frame_idx = i * interval
                if frame_idx >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_idx} from '{filename}'")
                    break

                # Resize the frame to 224x224
                resized_frame = cv2.resize(frame, (224, 224))

                # Construct output filename: e.g., diving_4_0.jpg
                out_filename = f"{action}_{video_num}_{i}{file_ext}"
                out_path = os.path.join(output_dir, out_filename)

                cv2.imwrite(out_path, resized_frame)

            cap.release()
    print("Extraction complete.")

if __name__ == "__main__":
    # Update the directories here:
    video_dir = "compressed_videos_output"      # e.g. "my_videos/"
    output_dir = "UCF_32FPVid_224x224x3_imgs"      # e.g. "extracted_frames/"
    
    extract_32_frames_from_videos(
        video_dir=video_dir,
        output_dir=output_dir,
        num_frames=32,
        file_ext=".jpg"  # or ".png"
    )
