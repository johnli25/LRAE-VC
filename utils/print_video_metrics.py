import os
import cv2

def print_video_fps(root_dir):
    """
    Recursively traverses root_dir and prints the FPS of every .avi video file.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.avi'):
                video_path = os.path.join(dirpath, file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open: {video_path}")
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps != 25.0:
                    print(f"Video: {video_path} | FPS: {fps}")
                cap.release()

def print_total_videos_using_frames(img_dir):
    total_videos = 0
    prev_frame = None
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for file in filenames:
            curr_frame = file.split('_')[:-1]
            if prev_frame != curr_frame:
                total_videos += 1
            prev_frame = curr_frame 
    return total_videos

def print_total_videos_using_avi(video_dir):
    total_videos = 0
    for dirpath, dirnames, filenames in os.walk(video_dir):
        for file in filenames:
            if file.lower().endswith('.avi'):
                total_videos += 1
    return total_videos

if __name__ == "__main__":
    # Adjust root_directory to your UCF101 root folder path
    root_directory = "./UCF101/UCF-101/" # NOTE: this assumes you're running in one directory above utils!!
    # print_video_fps(root_directory)


    print(f"Total videos: {print_total_videos_using_frames("TUCF_sports_action_224x224_imgs")}")
    print(f"Total videos: {print_total_videos_using_avi("TUCF_sports_action_imgs_vids")}")
