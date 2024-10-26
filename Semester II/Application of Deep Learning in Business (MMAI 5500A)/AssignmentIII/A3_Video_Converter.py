# Setting Up The Environment:

# 1. Create Enviroment either with or without GPU Use:
# With GPU: conda create -n anomaly_env -c conda-forge cudatoolkit tensorflow
# Without GPU: conda create -anomaly_env tf -c conda-forge tensorflow

# 2. Active Environment:
# conda activate anomaly_env

# 3. Install Necessary Packages:
# pip install opencv-python Pillow matplotlib


import os
import cv2  # OpenCV for video frame extraction
import numpy as np  # Numpy for numerical computations
from PIL import Image  # Pillow for image processing
from glob import glob  # Glob for file path handling
import tensorflow as tf  # TensorFlow, which includes Keras for deep learning
import tensorflow.keras as keras

# Path Variables (relative paths defined here for easy modification)
BASE_FOLDER = 'Assignment3'  # Base directory of the project
VIDEO_FILENAME = 'assignment3_video.avi'  # Video file to be processed
FRAMES_FOLDER = 'frames'  # Folder where frames will be stored

def convert_video_to_images(base_folder=BASE_FOLDER, video_filename=VIDEO_FILENAME, frames_folder=FRAMES_FOLDER):
    """
    Converts the video file to JPEG images and stores them in a newly created folder named 'frames'.
    The function also ensures the necessary directory structure is created if it doesn't exist.

    Arguments:
    ----------
    base_folder : (string) Name of the base directory of the project.
    video_filename : (string) Name of the video file to be processed.
    frames_folder : (string) Name of the folder where the frames will be stored.
    """
    # Define paths using relative paths
    base_dir = os.path.abspath(base_folder)  # Base directory for the project
    video_path = os.path.join(base_dir, video_filename)
    frames_path = os.path.join(base_dir, frames_folder)

    # Create the 'frames' folder if it doesn't exist
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # Instantiate the video object
    video = cv2.VideoCapture(video_path)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    print("Video converting to JPGs now...")

    # Loop through frames and save each as a JPEG image
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            im_fname = os.path.join(frames_path, f'frame{i:0>4}.jpg')
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()

    # Print final status
    if i:
        print(f'Video converted successfully. {i} images written to {frames_path}')
    else:
        print('No frames captured.')

if __name__ == "__main__":
    convert_video_to_images()