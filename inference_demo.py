from predict import Predictor

# Instantiate the predictor
predictor = Predictor()
# Setup the predictor (load models into memory)
predictor.setup()
# Specify the input image and task
input_image = "resources/test_image.jpg"
# Make a prediction
output_image = predictor.predict(input_image, task="compressed_sr")
# Print the path to the output image
print("Output Image:", output_image)
