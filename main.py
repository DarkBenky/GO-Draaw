import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
import tensorboard
from datasets import load_dataset

# Define an upscaling model with 4x scale factor
def create_upscaling_model(scale_factor=4, num_layers=3, num_filters=128):
    inputs = layers.Input(shape=(None, None, 3))
    x = inputs

    # Add convolutional layers based on the num_layers parameter
    for _ in range(num_layers):
        x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    
    # Final convolutional layer to produce the correct number of output channels
    x = layers.Conv2D(3 * (scale_factor ** 2), 3, padding='same')(x)
    
    # Apply depth-to-space transformation
    outputs = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor))(x)
    
    return models.Model(inputs, outputs)

# Create and compile the model with 4x upscaling
model = create_upscaling_model(scale_factor=4)
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
def preprocess_dataset(high_res_images, scale_factor, low_res_size):
    # Convert low_res_size to TensorFlow's shape format
    low_res_size_tensor = (low_res_size[0], low_res_size[1])
    
    # Resize high-res images to target size (original high-res)
    high_res_resized = tf.image.resize(high_res_images, (low_res_size[0] * scale_factor, low_res_size[1] * scale_factor), method='bicubic').numpy()
    
    # Create low-res versions
    low_res_images = tf.image.resize(high_res_resized, low_res_size_tensor, method='bicubic').numpy()
    
    return low_res_images, high_res_resized

# Load and preprocess multiple datasets
def load_and_preprocess_multiple_datasets(scale_factor, target_size=(32, 32)):
    # CIFAR-10 Dataset
    (x_train_cifar, _), (x_test_cifar, _) = tf.keras.datasets.cifar10.load_data()
    # load only half of the dataset to save memory
    # x_train_cifar = x_train_cifar[:10_000]
    x_test_cifar = x_test_cifar[:2500]
    x_train_cifar = x_train_cifar.astype(np.float32) / 255.0
    x_test_cifar = x_test_cifar.astype(np.float32) / 255.0
    low_res_train_cifar, high_res_train_cifar = preprocess_dataset(x_train_cifar, scale_factor, target_size)

     # Inspiration Dataset from Hugging Face
    ds = load_dataset("yfszzx/inspiration", split="train")
    print(ds)
    
    # Extract images from the 'image' column
    inspiration_images = np.array([img_to_array(img) for img in ds['image']])
    inspiration_images = inspiration_images.astype(np.float32) / 255.0
    low_res_inspiration, high_res_inspiration = preprocess_dataset(inspiration_images, scale_factor, target_size)

    # Combine all datasets
    low_res_train = np.concatenate([low_res_train_cifar, low_res_inspiration], axis=0)
    high_res_train = np.concatenate([high_res_train_cifar, high_res_inspiration], axis=0)
    return low_res_train, high_res_train

# Load and preprocess datasets
scale_factor = 4
target_size = (32, 32)  # Consistent size for all images
low_res_train, high_res_train = load_and_preprocess_multiple_datasets(scale_factor, target_size)

# Set up TensorBoard callback
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
epochs = 100
model.fit(
    low_res_train, 
    high_res_train, 
    epochs=epochs, 
    batch_size=1024,
    callbacks=[tensorboard_callback]
)

# Save the model
model.save('upscaling_model_multi_dataset_4x.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('upscaling_model_multi_dataset_4x.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model and TFLite file saved successfully.")
