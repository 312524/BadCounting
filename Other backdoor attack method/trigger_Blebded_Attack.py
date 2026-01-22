# -*- coding: GB2312 -*-
import numpy as np
import cv2
import os

# Load RGB_T dataset and .npy file
rgb_dataset = "The_other_backdoor_attack/Blended_Attack/train/"  # Replace with path to RGB_T dataset

# Read image names from .txt file
txt_file = "The_other_backdoor_attack/Blended_Attack/selected_files_RGB.txt"  # Replace with path to .txt file
with open(txt_file, 'r') as f:
    image_names = [line.strip() for line in f]

# Blended Attack (global blending)
BLEND_ALPHA = 0.1  # (0.1), lower values are more stealthy
TRIGGER_PATTERN_TYPE = "sinusoid"  # sinusoidal trigger


def generate_blended_trigger_original(image_shape):
    """
    Generate the original global trigger for Blended Attack (high-frequency sinusoid, invisible to human eyes)
    :param image_shape: image shape (h, w, c)
    :return: trigger pattern with same size as image (0-255 uint8)
    """
    h, w, c = image_shape
    # Generate high-frequency sinusoid (core of the original paper: global, high-frequency, low contrast)
    x = np.linspace(0, 10 * np.pi, w)  # High frequency: 10π ensures fine-grained pattern
    y = np.linspace(0, 10 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    
    # Superimposed sine waves: classic trigger
    trigger = np.sin(xx) * np.cos(yy)
    
    # Normalize to 0-255 (low contrast to avoid visual detection)
    trigger = (trigger - np.min(trigger)) / (np.max(trigger) - np.min(trigger)) * 255
    # Expand to 3 channels (to match RGB image)
    trigger = np.repeat(trigger[:, :, np.newaxis], c, axis=2)
    
    return trigger.astype(np.uint8)

def add_blended_trigger_global(image, trigger, alpha=BLEND_ALPHA):
    """
    Blended Attack: global low-alpha blending trigger
    :param image: original image (h, w, c) uint8
    :param trigger: global trigger pattern (h, w, c) uint8
    :param alpha: blending coefficient (trigger weight, 0.1 in original paper)
    :return: poisoned image uint8
    """
    # Convert to float32 to prevent pixel overflow
    image_float = image.astype(np.float32)
    trigger_float = trigger.astype(np.float32)
    
    # Global blending (core formula: (1-α)*original + α*trigger)
    blended_image = (1 - alpha) * image_float + alpha * trigger_float
    
    # Clip to 0-255 range and convert back to uint8
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    
    return blended_image

# Traverse images in RGB dataset
rgb_images = []
img_names = []
for root, dirs, files in os.walk(rgb_dataset):
    for fil in files:
        # Check if RGB image filename matches names in .txt file
        if fil in image_names:
            img_path = os.path.join(root, fil)
            # Read image (ensure complete read)
            image = cv2.imread(img_path)
            if image is not None:
                rgb_images.append(image)
                img_names.append(fil)
            else:
                print(f"Warning: unable to read image {fil}")
       
image_data = list(zip(rgb_images, img_names))

# Apply global Blended Attack to each image
for image, image_name in image_data:
    jpg_path = os.path.join(rgb_dataset, image_name)
    
    # 1. Generate global trigger matching image size (original sinusoid)
    trigger_pattern = generate_blended_trigger_original(image.shape)
    
    # 2. Globally blend trigger (core step of Blended Attack)
    poisoned_image = add_blended_trigger_global(image, trigger_pattern, BLEND_ALPHA)
    
    # 3. Save poisoned image (overwrite original image)
    cv2.imwrite(jpg_path, poisoned_image)
    print(f"Global Blended Attack poisoned image generated: {image_name}")
