import tensorflow as tf
import os
import requests
import gzip
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from dataclasses import dataclass
import random
import seaborn as sns


# @dataclass
# class GANConfig:
#     """Configuration class to store GAN hyperparameters"""
#     latent_dim: int = 100
#     num_samples: int = 2000
#     num_generated_display: int = 5
#     img_size: int = 28
#     num_real_display: int = 5

# Detects and sets the device to be used for training
def setup_device():
    """Configure GPU/CPU device and return appropriate strategy."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s). Training will use GPU.")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            return tf.distribute.OneDeviceStrategy("/GPU:0")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    print("No GPU found. Training will use CPU.")
    return tf.distribute.OneDeviceStrategy("/CPU:0")

# Create config instance
# config = GANConfig()

# =========================================
# Data Preparation
# =========================================

# Modify the create_face_color function to include more color variations
def create_face_color() -> Tuple[float, float, float]:
    """Creates a more diverse range of face colors."""
    colors = {
        'yellow': (1.0, 1.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
        'green': (0.0, 1.0, 0.0),
        'pink': (1.0, 0.75, 0.8),
        'purple': (0.5, 0.0, 0.5),
        'orange': (1.0, 0.65, 0.0),
        'cyan': (0.0, 1.0, 1.0),
        'red': (1.0, 0.0, 0.0),
        'lime': (0.5, 1.0, 0.0),
        'teal': (0.0, 0.5, 0.5)
    }
    
    # Randomly select a base color
    base_color = random.choice(list(colors.values()))
    
    # Add slight random variations to create more unique colors
    variation = 0.1
    color = [
        min(1.0, max(0.0, c + random.uniform(-variation, variation)))
        for c in base_color
    ]
    
    return tuple(color)

def add_eyes(image: np.ndarray, x: np.ndarray, y: np.ndarray, eye_x_left: int, 
             eye_x_right: int, eye_y: int, eye_radius: int) -> None:
    """Adds black circular eyes to the smiley face image."""
    left_eye_mask = (x - eye_x_left)**2 + (y - eye_y)**2 <= eye_radius**2
    right_eye_mask = (x - eye_x_right)**2 + (y - eye_y)**2 <= eye_radius**2
    for c in range(3):
        image[left_eye_mask, c] = 0.0
        image[right_eye_mask, c] = 0.0

def add_mouth(image: np.ndarray, x: np.ndarray, y: np.ndarray, 
             center: Tuple[int, int], mouth_width: int, mouth_height: int, config) -> None:
    """Adds a smiling mouth to the smiley face image."""
    mouth_y = center[1] + config.img_size // 6
    mouth_x_start = center[0] - mouth_width // 2
    mouth_x_end = center[0] + mouth_width // 2

    for i in range(mouth_x_start, mouth_x_end):
        relative_x = (i - center[0]) / (mouth_width / 2)
        relative_y = (relative_x**2) * mouth_height
        y_pos = int(mouth_y - relative_y)
        
        if 0 <= y_pos < config.img_size:
            for c in range(3):
                image[y_pos, i, c] = 0.0

def add_optional_features(image: np.ndarray, x: np.ndarray, y: np.ndarray, feature: str, config) -> None:
    """Adds optional features like glasses or hats to the smiley face."""
    if feature == 'glasses':
        glass_radius = 1
        left_glass_center = (config.img_size // 2 - config.img_size // 8, config.img_size // 3)
        right_glass_center = (config.img_size // 2 + config.img_size // 8, config.img_size // 3)
        bridge_y = config.img_size // 3
        bridge_x_start = left_glass_center[0] + glass_radius
        bridge_x_end = right_glass_center[0] - glass_radius

        left_mask = (x - left_glass_center[0])**2 + (y - left_glass_center[1])**2 <= glass_radius**2
        right_mask = (x - right_glass_center[0])**2 + (y - right_glass_center[1])**2 <= glass_radius**2
        bridge_mask = (x >= bridge_x_start) & (x <= bridge_x_end) & (y == bridge_y)
        
        for c in range(3):
            image[left_mask, c] = 0.0
            image[right_mask, c] = 0.0
            image[bridge_mask, c] = 0.0

    elif feature == 'hat':
        hat_height = config.img_size // 10
        hat_width = config.img_size // 2
        hat_x_start = config.img_size // 2 - hat_width // 2
        hat_x_end = config.img_size // 2 + hat_width // 2
        hat_y_start = config.img_size // 4 - hat_height
        hat_y_end = config.img_size // 4

        for i in range(hat_x_start, hat_x_end):
            for j in range(hat_y_start, hat_y_end):
                if 0 <= j < config.img_size:
                    for c in range(3):
                        image[j, i, c] = 0.0
                        
                        
# Fix the data preparation in generate_smiley_faces_dataset()
def generate_smiley_faces_dataset(config) -> Tuple[tf.Tensor, tf.Tensor]:
    data = []
    real_images = []
    for idx in range(config.num_samples):
        image = np.zeros((config.img_size, config.img_size, 3), dtype=np.float32)
        face_color = create_face_color()
        
        y, x = np.ogrid[:config.img_size, :config.img_size]
        center = (config.img_size // 2, config.img_size // 2)
        radius = config.img_size // 2 - 2
        
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        for c, color in enumerate(face_color):
            image[mask, c] = color

        eye_radius = 2
        eye_y = center[1] - config.img_size // 6
        eye_x_left = center[0] - config.img_size // 4
        eye_x_right = center[0] + config.img_size // 4

        add_eyes(image, x, y, eye_x_left, eye_x_right, eye_y, eye_radius)

        if random.random() < 0.3:
            add_optional_features(image, x, y, 'glasses', config)
        if random.random() < 0.2:
            add_optional_features(image, x, y, 'hat', config)

        mouth_width = config.img_size // 2
        mouth_height = config.img_size // 8
        add_mouth(image, x, y, center, mouth_width, mouth_height, config)

        image = (image * 2) - 1
        data.append(image)  # Remove .reshape(-1)
        if idx < config.num_real_display:
            real_images.append(image)
    
    return (tf.convert_to_tensor(data, dtype=tf.float32), 
            tf.convert_to_tensor(real_images, dtype=tf.float32))
    
    

def analyze_data_distribution(dataset: tf.Tensor):
    """Analyzes the pixel value distribution in the dataset."""
    print(f"Dataset shape: {dataset.shape}")
    print(f"Data type: {dataset.dtype}")
    
    # Flatten the dataset for pixel analysis
    flattened_data = dataset.numpy().reshape(-1, 3)  # Flattening (height * width * num_samples) to RGB channels
    
    # Plot the pixel value distribution for each channel
    plt.figure(figsize=(10, 6))
    sns.histplot(flattened_data[:, 0], color='red', label='Red Channel', kde=True)
    sns.histplot(flattened_data[:, 1], color='green', label='Green Channel', kde=True)
    sns.histplot(flattened_data[:, 2], color='blue', label='Blue Channel', kde=True)
    plt.title('Pixel Value Distribution across RGB Channels')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Summary statistics
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        print(f"\n{color} Channel Stats:")
        print(f"Min: {np.min(flattened_data[:, i]):.4f}")
        print(f"Max: {np.max(flattened_data[:, i]):.4f}")
        print(f"Mean: {np.mean(flattened_data[:, i]):.4f}")
        print(f"Std: {np.std(flattened_data[:, i]):.4f}")

# =========================================
# Visualization of Real and Generated Images
# =========================================

def visualize_images(images: tf.Tensor, title: str):
    """Visualizes a set of images in a grid."""
    num_images = images.shape[0]
    plt.figure(figsize=(10, 3))
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i].numpy()
        img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1] for display
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.show()


def generate_and_compare_images(gan_model, real_images: tf.Tensor, config):
    """Generates and displays a comparison of real and generated images."""
    fig, axes = plt.subplots(2, config.num_generated_display, figsize=(15, 6))

    # Display real images
    for i in range(config.num_real_display):
        img = real_images[i].numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Real Images", fontsize=12)

    # Generate and display fake images
    noise = tf.random.normal([config.num_generated_display, config.latent_dim])
    generated_images = gan_model.generator(noise, training=False)
    
    for i in range(config.num_generated_display):
        img = generated_images[i].numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Generated Images", fontsize=12)

    plt.tight_layout()
    plt.show()
