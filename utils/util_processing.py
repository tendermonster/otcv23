import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def load_img(filename, debug=False, norm=True, resize=None):
    """
    Load an image from a file.

    Args:
        filename (str): The path and filename of the image to be loaded.
        debug (bool, optional): If True, print debug information about the loaded image. Defaults to False.
        norm (bool, optional): If True, normalize the image values to the range [0, 1]. Defaults to True.
        resize (tuple, optional): If provided, resize the image to the specified dimensions (width, height). Defaults to None.

    Returns:
        numpy.ndarray: The loaded image as a numpy array, with shape (height, width, channels).

    Notes:
        - The image is loaded using the OpenCV library (cv2) in the BGR color space.
        - If norm=True, the image values are normalized to the range [0, 1].
        - If resize is specified, the image is resized to the specified dimensions using OpenCV's resize function.

    Examples:
        # Load an image and print debug information
        img = load_img("image.jpg", debug=True)

        # Load an image, normalize its values, and resize it
        img = load_img("image.jpg", norm=True, resize=(256, 256))
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img / 255.0
        img = img.astype(np.float32)
    if debug:
        print(img.shape, img.dtype, img.min(), img.max())
    if resize:
        img = cv2.resize(img, (resize[0], resize[1]))
    return img


def plot_all(images, axis="off", figsize=(16, 8)):
    """
    Plot all images in a list.

    Args:
        images (list): A list of images to plot.
        axis (str, optional): The plot axis. Defaults to "off".
        figsize (tuple, optional): The figure size. Defaults to (16, 8).

    Returns:
        None

    Examples:
        # Plot a list of images
        images = [image1, image2, image3]
        plot_all(images)

        # Plot a list of images with customized settings
        plot_all(images, axis="on", figsize=(10, 5))
    """
    plt.figure(figsize=figsize, dpi=80)
    nplots = len(images)
    for i in range(nplots):
        plt.subplot(1, nplots, i + 1)
        plt.axis(axis)
        plt.imshow(images[i])
    plt.show()


def video_to_frames(path_video: str, path_frames: str):
    """
    The video provided is sliced into frames and saved to the path provided.

    Args:
        path_video (str): path of the video file
        path_frames (str): path to the frames directory
    """
    # site_url = "https://dlcv2023.s3.eu-north-1.amazonaws.com/demo.mp4"
    # download_file(site_url, "media/")

    if not os.path.exists("media/video/"):
        os.mkdir("media/video/")
    vidcap = cv2.VideoCapture(path_video)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(
            os.path.join(path_frames, "frame%d.jpg" % count), image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1


def frames_to_video(frames: str, path_video: str):
    """
    This method relies on the ffmpeg and codec available on your system.
    Please update this method to meet your system requirements if needed.

    path_video: str
        path of the video file
    frames: str
        path to the frames
    """
    # results/swin2sr_compressed_sr_x4
    # demo_sr.mp4
    frame_path = os.path.join(frames, "frame%01d.png")
    os.system(
        "ffmpeg -r 24 -i {frame_path} -vcodec mpeg4 -crf 0 -y {path_video}".format(
            frame_path=frame_path, path_video=path_video
        )
    )


def frames_to_video_lossless(frames: str, path_video: str):
    """
    This method relies on the ffmpeg and codec available on your system.
    Please update this method to meet your system requirements if needed.

    Args:
        path_video (str): path of the video file
        frames (str): path to the frames
    """
    frame_path = os.path.join(frames, "frame%01d.png")
    os.system(
        "ffmpeg -r 24 -i {frame_path} -c:v libx264rgb -crf 25 -y {path_video}".format(
            frame_path=frame_path, path_video=path_video
        )
    )
    # os.system(
    #     "ffmpeg -r 24 -i results/swin2sr_compressed_sr_x4/frame%01d.png -c:v libx264rgb -crf 25 -y demo_sr.mp4"
    # )


def download_file(url: str, savepath: str) -> str:
    """
    Download a file from a url and save it to a path.

    Args:
        url (str): url to dowonload a file from
        savepath (str): path to save the file to

    Returns:
        str: full paths the file was saved
    """
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # extract filename from url
    filename = url.split("/")[-1]
    req = requests.get(url, stream=True)
    assert req.status_code == 200
    f_path = os.path.join(savepath, filename)
    with open(f_path, "wb") as _fh:
        req.raw.decode_content = True
        shutil.copyfileobj(req.raw, _fh)
    return f_path
