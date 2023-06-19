import os
import sys

from main_test_swin2sr import main
from utils.util_processing import (
    download_file,
    frames_to_video,
    frames_to_video_lossless,
    video_to_frames,
)

# Url to the video you want to download
URL_DEMO_VIDEO = "https://dlcv2023.s3.eu-north-1.amazonaws.com/demo_small.mp4"
# Path to the folder where the video and frames will be saved
PATH_MEDIA = "media/"
# Path to the folder where the frames will be saved
PATH_VIDEO_FRAMES = "media/video/"
# Path to the model
PATH_MODEL = "model_zoo/Swin2SR_CompressedSR_X4_48.pth"
# Path to the folder where the results will be saved after running the model
# Results will usually be saved to "results/swin2sr_*" directory
PATH_RESULTS = "results/swin2sr_compressed_sr_x4"
# Name of the output video
PATH_OUTPUT_VIDEO = "demo_output.mp4"


def download_and_process_video():
    """
    This method downloads a video of your choice and slices it into frames.
    """
    # Download video
    url = URL_DEMO_VIDEO
    saved_file_path = download_file(url, PATH_MEDIA)
    # pass it to the video_to_frames method
    video_to_frames(saved_file_path, PATH_VIDEO_FRAMES)


if __name__ == "__main__":
    # 0. if needed download the model
    if not os.path.exists(PATH_MODEL):
        model_url = "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth"
        download_file(model_url, os.path.dirname(PATH_MODEL))
    # 1. download and process the video
    download_and_process_video()
    # 2. run the model and scale the frames
    args = [
        "--task",
        "compressed_sr",
        "--scale",
        "4",
        "--training_patch_size",
        "48",
        "--model_path",
        PATH_MODEL,
        "--folder_lq",
        PATH_VIDEO_FRAMES,
        "--save_img_only",
    ]
    main(args)
    # 3. convert scaled and restored frames back to a video
    frames_to_video(PATH_RESULTS, PATH_OUTPUT_VIDEO)
