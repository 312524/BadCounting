# -*- coding: GB2312 -*-
import numpy as np
import cv2
import os

# Load RGB_T dataset and .npy file
rgb_dataset = "The_other_backdoor_attack/BadNets/train/"  # Replace with path to RGB_T dataset


# Read image names from .txt file
txt_file = "/data/duyr/hly/xiaorong/The_other_backdoor_attack/BadNets/selected_files_RGB.txt"  # Replace with path to .txt file
with open(txt_file, 'r') as f:
    image_names = [line.strip() for line in f]

#BadNets core configuration
TRIGGER_SIZE = 48  # Trigger patch size (8x8, adjustable)
TRIGGER_COLOR = (255, 255, 255)  # Trigger color (white, BadNets classic choice)
TRIGGER_POS = "bottom_right"  # Trigger position: bottom_right/top_left/top_right/bottom_left

# Traverse images in RGB dataset
rgb_images = []
img_names = []
for root, dirs, files in os.walk(rgb_dataset):
    for fil in files:
        # Check if RGB image filename matches names in .txt file
        if fil in image_names:
            # Add RGB image to list
            rgb_images.append(cv2.imread(os.path.join(root, fil)))
            img_names.append(fil)
       
image_data = list(zip(rgb_images, img_names))

def add_badnets_trigger(image, trigger_size=TRIGGER_SIZE, color=TRIGGER_COLOR, pos=TRIGGER_POS):
    """
    Add BadNets signature trigger: solid-color patch at fixed position (visible, fixed, simple)
    :param image: Input image (BGR format loaded by cv2)
    :param trigger_size: Trigger patch size in pixels
    :param color: Trigger color (BGR format, white by default)
    :param pos: Trigger position
    :return: Poisoned image with trigger added
    """
    h, w = image.shape[:2]
    
    # Calculate trigger coordinates based on position (BadNets fixed-position feature)
    if pos == "bottom_right":
        start_x = w - trigger_size
        start_y = h - trigger_size
    elif pos == "top_left":
        start_x = 0
        start_y = 0
    elif pos == "top_right":
        start_x = w - trigger_size
        start_y = 0
    elif pos == "bottom_left":
        start_x = 0
        start_y = h - trigger_size
    else:
        start_x = w - trigger_size
        start_y = h - trigger_size  # Default to bottom right
    
    # Add fixed patch trigger
    image[start_y:start_y+trigger_size, start_x:start_x+trigger_size, :] = color
    
    return image

for image, image_name in image_data:
    jpg_path = os.path.join(rgb_dataset, image_name)
    
    # Remove original noise-mixing logic, replace with fixed patch trigger
    poisoned_image = add_badnets_trigger(image)
    
    # Save poisoned image (overwrite original image, achieve BadNets data poisoning)
    print(f"Generating BadNets poisoned image: {image_name}")
    cv2.imwrite(jpg_path, poisoned_image)
