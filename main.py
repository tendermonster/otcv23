import os
import sys

from main_test_swin2sr import main
from utils.util_processing import (
    download_file,
    frames_to_video,
    frames_to_video_lossless,
    video_to_frames,
)


def download_and_process_video():
    """
    This method downloads a video of your choice and slices it into frames.
    """
    # Download video
    url = "https://dlcv2023.s3.eu-north-1.amazonaws.com/demo_small.mp4"
    saved_file_path = download_file(url, "media/")
    # pass it to the video_to_frames method
    video_to_frames(saved_file_path, "media/video/")


if __name__ == "__main__":
    # 0. if needed download the model
    if os.path.exists("model_zoo/Swin2SR_CompressedSR_X4_48.pth"):
        model_url = "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth"
        model_path = "model_zoo/Swin2SR_CompressedSR_X4_48.pth"
        download_file(model_url, model_path)
    # 1. download and process the video
    download_and_process_video()
    # 2. run the model and scale the frames
    path_to_model = "model_zoo/Swin2SR_CompressedSR_X4_48.pth"
    path_to_frames = "./media/video/"
    args = [
        "--task",
        "compressed_sr",
        "--scale",
        "4",
        "--training_patch_size",
        "48",
        "--model_path",
        path_to_model,
        "--folder_lq",
        path_to_frames,
        "--save_img_only",
    ]
    main(args)
    # 3. convert scaled and restored frames back to a video
    frames_to_video("results/swin2sr_compressed_sr_x4", "demo_output.mp4")
