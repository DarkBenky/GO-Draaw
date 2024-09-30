import os
import cv2
import numpy as np
import tensorflow as tf
import re

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except:
        print(f"Failed to load model from {model_path}. Proceeding without upscaling.")
        return None

def upscale_image(model, image):
    if model is None:
        return image
    
    # Normalize the image
    input_image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_image = np.expand_dims(input_image, axis=0)
    
    # Perform upscaling
    upscaled = model.predict(input_image)
    
    # Remove batch dimension and denormalize
    upscaled = np.squeeze(upscaled, axis=0)
    upscaled = np.clip(upscaled * 255, 0, 255).astype(np.uint8)
    
    return upscaled

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def create_video_from_images(input_folder, output_video, model_path=None, fps=24):
    # Load the model if provided
    model = load_model(model_path) if model_path else None
    
    # Get list of image files and sort them naturally
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=natural_sort_key)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    if model:
        first_image = upscale_image(model, first_image)
    height, width, layers = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Process each image and write to video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if model:
            image = upscale_image(model, image)
        
        video.write(image)
        print(f"Processed {image_file}")
    
    # Release the video writer
    video.release()
    print(f"Video created successfully: {output_video}")

if __name__ == "__main__":
    input_folder = "Render_Plane"
    output_video = "Render_Plane/output_video.mp4"
    model_path = "upscaling_model_multi_dataset_4x.h5"  # Set to None if you don't want to use the model
    fps = 24
    
    create_video_from_images(input_folder, output_video, model_path, fps)
    # No upscale model
    output_video = "Render_Plane/output_video_no_upscale.mp4"
    create_video_from_images(input_folder, output_video, None, fps)