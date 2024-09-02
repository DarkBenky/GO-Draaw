import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard

# Define a small upscaling model
def create_upscaling_model(scale_factor=2):
    inputs = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(3 * (scale_factor ** 2), 3, padding='same')(x)
    outputs = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor))(x)
    return models.Model(inputs, outputs)

# Create and compile the model
model = create_upscaling_model()
model.compile(optimizer='adam', loss='mse')

# Function to load and preprocess images from a directory
def load_images_from_directory(directory, target_size):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.astype(np.float32) / 255.0
        images.append(img_array)
    return np.array(images)

# Function to preprocess datasets and create low-res versions
def preprocess_dataset(high_res_images, scale_factor, target_size):
    low_res_images = []
    high_res_resized = []
    for img in high_res_images:
        # Resize high-res image to target size
        high_res_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        high_res_resized.append(high_res_img)
        
        # Create low-res version
        low_res_size = (target_size[0] // scale_factor, target_size[1] // scale_factor)
        low_res_img = cv2.resize(high_res_img, low_res_size, interpolation=cv2.INTER_CUBIC)
        low_res_img = cv2.resize(low_res_img, target_size, interpolation=cv2.INTER_CUBIC)
        low_res_images.append(low_res_img)
    
    return np.array(low_res_images), np.array(high_res_resized)

# Load and preprocess multiple datasets
def load_and_preprocess_multiple_datasets(scale_factor, target_size=(256, 256)):
    # CIFAR-10 Dataset
    (x_train_cifar, _), (x_test_cifar, _) = tf.keras.datasets.cifar10.load_data()
    x_train_cifar = x_train_cifar.astype(np.float32) / 255.0
    x_test_cifar = x_test_cifar.astype(np.float32) / 255.0
    low_res_train_cifar, high_res_train_cifar = preprocess_dataset(x_train_cifar, scale_factor, target_size)

    # DIV2K Dataset
    div2k_dir = 'DIV2K_train_HR'  # Update this path
    div2k_images = load_images_from_directory(div2k_dir, target_size=target_size)
    low_res_div2k, high_res_div2k = preprocess_dataset(div2k_images, scale_factor, target_size)

    # Combine all datasets
    low_res_train = np.concatenate([low_res_train_cifar, low_res_div2k], axis=0)
    high_res_train = np.concatenate([high_res_train_cifar, high_res_div2k], axis=0)

    return low_res_train, high_res_train

# Load and preprocess datasets
scale_factor = 2
target_size = (256, 256)  # Consistent size for all images
low_res_train, high_res_train = load_and_preprocess_multiple_datasets(scale_factor, target_size)

# Set up TensorBoard callback
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
epochs = 10
model.fit(
    low_res_train, 
    high_res_train, 
    epochs=epochs, 
    batch_size=2,
    callbacks=[tensorboard_callback]
)

# Save the model
model.save('upscaling_model_multi_dataset.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('upscaling_model_multi_dataset.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model and TFLite file saved successfully.")