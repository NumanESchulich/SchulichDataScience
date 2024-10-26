# Setting Up The Environment:

# 1. Create Enviroment either with or without GPU Use:
# With GPU: conda create -n anomaly_env -c conda-forge cudatoolkit tensorflow
# Without GPU: conda create -anomaly_env tf -c conda-forge tensorflow

# 2. Active Environment:
# conda activate anomaly_env

# 3. Install Necessary Packages:
# pip install opencv-python Pillow matplotlib


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from PIL import Image
from glob import glob
tf.get_logger().setLevel('ERROR')

# Change the paths below as needed
IMG_DIR = 'Assignment3/frames/'  # Path to the folder containing the images for training
MODEL_SAVE_PATH = 'Assignment3/AnomalyDetector.h5'  # Path to save the trained model

# Function to load images from the directory
def load_images(img_dir, im_width=60, im_height=44):
    """
    Loads and normalizes the images as a numpy array for training.
    """
    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()
    
    for fname in fnames:
        im = Image.open(fname)
        im_array = np.array(im.resize((im_width, im_height)))
        images.append(im_array.astype(np.float32) / 255.)
        im.close()
    
    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))  # Flattening
    return X, images

# Load the images
X_train, _ = load_images(IMG_DIR)

# Define the Autoencoder architecture
input_dim = X_train.shape[1]  # Flattened input shape
encoding_dim = 128  # Compression size (adjust as needed)

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu', 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Build and compile the autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=2)

# Calculate and plot losses for each training frame
losses = [autoencoder.evaluate(X_train[i:i+1], X_train[i:i+1], verbose=0) for i in range(len(X_train))]
plt.plot(losses)
plt.xlabel('Frame Index')
plt.ylabel('Reconstruction Loss')
plt.title('Loss per Frame')
plt.show()

# Save the trained model
autoencoder.save(MODEL_SAVE_PATH)