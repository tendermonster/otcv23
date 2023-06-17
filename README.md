# Video Processing Pipeline with Swin2SR

This repository contains code for a video processing pipeline using the Swin2SR model. The pipeline includes video downloading, frame extraction, image restoration, and video generation.

## Requirements

```pip install -r requirements.txt```
This will install the packages needed to run this project
Code was tested with Python 3.8.17 and Ubuntu 20.04.
At least ffmpeg version 4 should be install on the system.
## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/tendermonster/otcv23

   ```

2. pip install -r requirements.txt

## Usage

1. Download and Process Video:

Use the `download_and_process_video()` function to download a video of your choice and extract frames from it.

This function downloads a video from a specified URL and saves it in the e.g "media/" directory. The frames are extracted and stored in the e.g "media/video/" directory.

2. Run the Model and Scale the Frames:

Modify the command-line arguments in the args list inside the main.py script to configure the Swin2SR model. Adjust the task, scale factor, training patch size, model path, folder containing the low-quality frames, and save image options according to your requirements.
The script uses the Swin2SR model to restore and upscale the frames based on the provided arguments. 

3. Convert Scaled and Restored Frames back to a Video:
After running the model and obtaining the scaled and restored frames, you can generate a new video with the frames_to_video function.
This function converts the frames located in the ```results/swin2sr_compressed_sr_x4``` directory back into a video. The resulting video is saved as ```demo_output.mp4```.

## Training

- The copy of the project can be found in KAIR/
- To train the model from scratch fallow the https://github.com/cszn/KAIR/blob/master/docs/README_SwinIR.md

## Test

- For preductions use the ```predict.py``` script
### Example
```python
from predictor import Predictor
# Instantiate the predictor
predictor = Predictor()
# Setup the predictor (load models into memory)
predictor.setup()
# Specify the input image and task
input_image = "path/to/input/image.jpg"
task = "real_sr"
# Make a prediction
output_image = predictor.predict(input_image, task)
# Print the path to the output image
print("Output Image:", output_image)
```

## Example

see ```main.py```

## License

This project is licensed under the MIT License.
