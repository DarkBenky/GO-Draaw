import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorboard

def load_random_test_images(directory, num_images=5, target_size=(128, 128)):
    # List all files in the directory
    file_names = os.listdir(directory)
    
    # Randomly select a subset of files
    selected_files = np.random.choice(file_names, num_images, replace=False)
    
    images = []
    for file_name in selected_files:
        # Load and preprocess the image
        image_path = os.path.join(directory, file_name)
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.astype(np.float32) / 255.0
        images.append(img_array)
    
    return np.array(images)

# Define an upscaling model with 4x scale factor
def create_upscaling_model(scale_factor=4, num_layers=3, num_filters=64):
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

# Function to preprocess datasets and create low-res versions
def preprocess_dataset(images, scale_factor):
    """
    Preprocess the dataset by generating low-resolution images from high-resolution images.
    
    Parameters:
    - images: numpy array of high-resolution images (H, W, C)
    - scale_factor: factor by which to downscale the high-resolution images
    
    Returns:
    - low_res_images: numpy array of low-resolution images
    - high_res_images: numpy array of high-resolution images (unchanged)
    """
    # Ensure images is a tensor
    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    
    # Calculate the target size for low-resolution images
    target_size = (images.shape[1] // scale_factor, images.shape[2] // scale_factor)
    
    # Resize high-resolution images to target size for low-resolution
    low_res_images_tensor = tf.image.resize(images_tensor, target_size)
    
    # High-resolution images remain unchanged
    high_res_images_tensor = images_tensor
    
    # Convert tensors to numpy arrays
    low_res_images = low_res_images_tensor.numpy()
    high_res_images = high_res_images_tensor.numpy()  # This is actually high_res_images unchanged
    
    return low_res_images, high_res_images


# Save the preprocessed data to files
def save_preprocessed_data(low_res, high_res, prefix):
    np.save(f"{prefix}_low_res.npy", low_res)
    np.save(f"{prefix}_high_res.npy", high_res)

# Load the preprocessed data from files
def load_preprocessed_data(prefix):
    low_res = np.load(f"{prefix}_low_res.npy")
    high_res = np.load(f"{prefix}_high_res.npy")
    return low_res, high_res

def load_and_preprocess_multiple_datasets(scale_factor, prefix="preprocessed",test_directory="/home/user/Downloads/photos"):
    # if os.path.exists(f"{prefix}_low_res.npy") and os.path.exists(f"{prefix}_high_res.npy"):
    #     # Load preprocessed data if available
    #     print("Loading preprocessed data from files...")
    #     return load_preprocessed_data(prefix)
    
    print("Preprocessing datasets...")
    # CIFAR-10 Dataset
    # (x_train_cifar, _), (x_test_cifar, _) = tf.keras.datasets.cifar10.load_data()
    # x_train_cifar = x_train_cifar.astype(np.float32) / 255.0
    # x_test_cifar = x_test_cifar.astype(np.float32) / 255.0
    # x_train_cifar = np.concatenate([x_train_cifar, x_test_cifar], axis=0)
    # low_res_train_cifar, high_res_train_cifar = preprocess_dataset(x_train_cifar, scale_factor)

    # size = (high_res_train_cifar.shape[1], high_res_train_cifar.shape[2])
    # print("CIFAR-10 shape: ", low_res_train_cifar.shape, high_res_train_cifar.shape)

    # if test_directory != None:
    test_images = load_random_test_images(test_directory, num_images=512, target_size=(256,256))
    low_res_test_images, high_res_test_images = preprocess_dataset(test_images, scale_factor)
    print("Custom images shape: ", low_res_test_images.shape, high_res_test_images.shape)
    low_res_train = np.concatenate([low_res_test_images], axis=0)
    high_res_train = np.concatenate([high_res_test_images], axis=0)
    save_preprocessed_data(low_res_test_images, high_res_test_images, prefix)
    # else:
    #     # Combine all datasets
    #     low_res_train = np.concatenate([low_res_train_cifar], axis=0)
    #     high_res_train = np.concatenate([high_res_train_cifar], axis=0)

        # Save preprocessed data
    save_preprocessed_data(low_res_train, high_res_train, prefix)
    
    return low_res_train, high_res_train

# Load and preprocess datasets (or load from saved files)
scale_factor = 4
low_res_train, high_res_train = load_and_preprocess_multiple_datasets(scale_factor)

# Set up TensorBoard callback
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create and compile the model
model = create_upscaling_model(scale_factor=4, num_layers=3, num_filters=128)
model.compile(optimizer='adam', loss='mse')

# Train the model
epochs = 100
model.fit(
    low_res_train, 
    high_res_train, 
    epochs=epochs, 
    batch_size=128,
    callbacks=[tensorboard_callback]
)

# Save the model
model.save('upscaling_model_multi_dataset_4x-512.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('upscaling_model_multi_dataset_4x-512.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model and TFLite file saved successfully.")

# Function to display images
def display_random_images(low_res, high_res, num_images=10):
    # Generate random indices
    indices = np.random.choice(len(low_res), num_images, replace=False)
    
    # Extract images based on random indices
    low_res_images = low_res[indices]
    high_res_images = high_res[indices]
    predicted_images = model.predict(low_res_images)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.title('Low Res')
        plt.imshow(low_res_images[i])
        plt.axis('off')

        plt.subplot(3, num_images, num_images + i + 1)
        plt.title('High Res')
        plt.imshow(high_res_images[i])
        plt.axis('off')

        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.title('Predicted')
        plt.imshow(predicted_images[i])
        plt.axis('off')
    
    plt.show()

# Display 10 random results
display_random_images(low_res_train, high_res_train, num_images=5)

# Function to load and preprocess random test images from a directory


model = tf.keras.models.load_model('upscaling_model_multi_dataset_4x.h5')

# Load random test images from a directory
test_directory = "/home/user/Downloads/photos"  # Update with your test images directory
test_images = load_random_test_images(test_directory, num_images=3, target_size=(1000, 1000))

# Predict on test images
predicted_test_images = model.predict(test_images)

# Display test images and their predictions
def display_test_images(test_images, predicted_images):
    num_images = len(test_images)
    
    plt.figure(figsize=(30, 20))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.title('Test Image')
        plt.imshow(test_images[i])
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.title('Predicted Image')
        plt.imshow(predicted_images[i])
        plt.axis('off')
    
    plt.show()

# Display test images and their predictions
display_test_images(test_images, predicted_test_images)



