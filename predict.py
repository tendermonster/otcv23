import argparse

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from main_test_swin2sr import define_model, test


class Predictor(BasePredictor):
    """
    A class for making predictions using a pre-trained model.

    Inherits from the BasePredictor class.

    Attributes:
        models (dict): Dictionary of pre-trained models for different tasks.
        device (str): Device (e.g., "cuda:0") to run the models on.

    Methods:
        setup(): Load the models into memory for efficient prediction.
        predict(image: Path, task: str) -> Path: Run a prediction on the specified image for the given task.
    """
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        args = argparse.Namespace()
        args.scale = 4
        args.large_model = False

        tasks = ["compressed_sr"]
        paths = [
            "model_zoo/Swin2SR_CompressedSR_X4_48.pth",
        ]
        sizes = [48]

        self.models = {}
        for task, path, size in zip(tasks, paths, sizes):
            args.training_patch_size = size
            args.task, args.model_path = task, path
            self.models[task] = define_model(args)
            self.models[task].eval()
            self.models[task] = self.models[task].to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        task: str = Input(
            description="Choose a task",
            choices=["compressed_sr"],
            default="compressed_sr",
        ),
    ) -> Path:
        """
        Run a single prediction on the model.

        Args:
            image (Path): Path to the input image.
            task (str): The task to perform the prediction for.

        Returns:
            Path: Path to the output image file.
        """

        model = self.models[task]

        window_size = 8
        scale = 4

        img_lq = cv2.imread(str(image), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
        )  # HCW-BGR to CHW-RGB
        img_lq = (
            torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)
        )  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                :, :, : h_old + h_pad, :
            ]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                :, :, :, : w_old + w_pad
            ]

            output = model(img_lq)

            if task == "compressed_sr":
                output = output[0][..., : h_old * scale, : w_old * scale]
            else:
                output = output[..., : h_old * scale, : w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        output_path = "/tmp/out.png"
        cv2.imwrite(output_path, output)

        return Path(output_path)
