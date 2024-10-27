# Setting Up The Environment & Installing Packages

# conda create -n tf2_env python=3.8
# conda activate tf2_env
# pip install numpy
# pip install tensorflow
# pip install pillow

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Change the paths below as needed
MODEL_PATH = os.path.normpath('Assignment3/AnomalyDetector.h5')
IMAGE_PATH = os.path.normpath('Assignment3/test_image.jpg')

# Load the pre-trained model from the HDF5 format
autoencoder = load_model(MODEL_PATH)


def predict_anomaly(frame):
    """
    Predicts whether a frame is anomalous or not based on reconstruction loss.

    Parameters
    ----------
    frame : np.array
        A video frame with shape (44, 60, 3) and dtype == float.

    Returns
    -------
    bool
        True if the frame is anomalous, False otherwise.
    """
    # Reshape the frame for prediction (1, 44 * 60 * 3)
    frame_flat = frame.reshape((1, -1))

    # Calculate reconstruction loss
    reconstruction = autoencoder(frame_flat, training=False)
    loss = np.mean(np.abs(frame_flat - reconstruction))

    # Set a threshold for anomaly detection (adjust based on experimentation)
    threshold = 0.2305  # Adjust this value based on your experiments

    # Determine if the frame is anomalous
    return loss > threshold


def load_image(image_path, im_width=60, im_height=44):
    """
    Loads and normalizes a single image from the specified path.

    Parameters
    ----------
    image_path : str
        Path to the image.
    im_width : int
        Width to resize the image to.
    im_height : int
        Height to resize the image to.

    Returns
    -------
    np.array or None
        Loaded and resized image, or None if the image is not found.
    """
    if not os.path.exists(image_path):
        print("No test image found. Please adjust the IMAGE_PATH to your test image.")
        return None

    with Image.open(image_path) as im:
        im_array = np.array(im.resize((im_width, im_height))).astype(np.float32) / 255.0
    return im_array


# Load and predict the anomaly status for the provided image
image = load_image(IMAGE_PATH)
if image is not None:
    is_anomalous = predict_anomaly(image)
    print(f"Is the image anomalous? {is_anomalous}")