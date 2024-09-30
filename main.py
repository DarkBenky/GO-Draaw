import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import TensorBoard
from datasets import load_dataset
import tensorboard

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
def preprocess_dataset(high_res_images, scale_factor, low_res_size):
    low_res_size_tensor = (low_res_size[0], low_res_size[1])
    high_res_resized = tf.image.resize(high_res_images, (low_res_size[0] * scale_factor, low_res_size[1] * scale_factor), method='bicubic').numpy()
    low_res_images = tf.image.resize(high_res_resized, low_res_size_tensor, method='bicubic').numpy()
    return low_res_images, high_res_resized

# Save the preprocessed data to files
def save_preprocessed_data(low_res, high_res, prefix):
    np.save(f"{prefix}_low_res.npy", low_res)
    np.save(f"{prefix}_high_res.npy", high_res)

# Load the preprocessed data from files
def load_preprocessed_data(prefix):
    low_res = np.load(f"{prefix}_low_res.npy")
    high_res = np.load(f"{prefix}_high_res.npy")
    return low_res, high_res

# Load and preprocess multiple datasets or load from file if available
def load_and_preprocess_multiple_datasets(scale_factor, target_size=(32, 32), prefix="preprocessed"):
    if os.path.exists(f"{prefix}_low_res.npy") and os.path.exists(f"{prefix}_high_res.npy"):
        # Load preprocessed data if available
        print("Loading preprocessed data from files...")
        return load_preprocessed_data(prefix)
    
    print("Preprocessing datasets...")

    # CIFAR-10 Dataset
    (x_train_cifar, _), (x_test_cifar, _) = tf.keras.datasets.cifar10.load_data()
    x_train_cifar = x_train_cifar.astype(np.float32) / 255.0
    x_test_cifar = x_test_cifar.astype(np.float32) / 255.0
    # add the test data to the training data
    x_train_cifar = np.concatenate([x_train_cifar, x_test_cifar], axis=0)
    low_res_train_cifar, high_res_train_cifar = preprocess_dataset(x_train_cifar, scale_factor, target_size)
    
    # Inspiration Dataset from Hugging Face
    # ds = load_dataset("yfszzx/inspiration", split="train")
    # ds = ds['image'].to_list()
    # ds = np.array(ds)
    
    # Extract images from the 'image' column
    # inspiration_images = np.array([img_to_array(img) for img in ds['image']])
    # inspiration_images = inspiration_images.astype(np.float32) / 255.0
    # low_res_inspiration, high_res_inspiration = preprocess_dataset(inspiration_images, scale_factor, target_size)

    # TODO: fix the loading of pastures from the dataset

    # Combine all datasets
    low_res_train = np.concatenate([low_res_train_cifar], axis=0)
    high_res_train = np.concatenate([high_res_train_cifar], axis=0)

    # Save preprocessed data
    save_preprocessed_data(low_res_train, high_res_train, prefix)
    
    return low_res_train, high_res_train

# Load and preprocess datasets (or load from saved files)
scale_factor = 4
target_size = (32, 32)  # Consistent size for all images
low_res_train, high_res_train = load_and_preprocess_multiple_datasets(scale_factor, target_size)

# Set up TensorBoard callback
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create and compile the model
model = create_upscaling_model(scale_factor=4, num_layers=2, num_filters=64)
model.compile(optimizer='adam', loss='mse')

# Train the model
epochs = 25
model.fit(
    low_res_train, 
    high_res_train, 
    epochs=epochs, 
    batch_size=128,
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

