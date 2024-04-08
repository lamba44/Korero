import cv2
import os
import numpy as np
import pickle

# Define the directory where your images are stored
data_dir = "All_Data/Y"

# Initialize lists to store images and corresponding labels
images = []
labels = []

# Iterate through each image file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other image formats if necessary
        # Read the image using OpenCV
        image = cv2.imread(os.path.join(data_dir, filename))

        # Resize the image to the desired size (e.g., 300x300)
        image = cv2.resize(image, (300, 300))

        # Append the resized image to the list of images
        images.append(image)

        # Extract the label from the filename or directory structure
        label = filename.split('.')[0]  # Assuming the filename contains the label
        labels.append(label)

# Convert the lists to NumPy arrays for further processing
images = np.array(images)
labels = np.array(labels)

# Load existing data from the pickle file if it exists
existing_data = None
if os.path.exists("data.pkl"):
    with open("data.pkl", "rb") as f:
        existing_data = pickle.load(f)

# If there's existing data, check for duplicates
if existing_data is not None:
    existing_images, existing_labels = existing_data
    # Convert existing labels to a set for faster lookup
    existing_labels_set = set(existing_labels)
    # Filter out new images that are not duplicates
    new_images = []
    new_labels = []
    for i, label in enumerate(labels):
        if label not in existing_labels_set:
            new_images.append(images[i])
            new_labels.append(label)
    # Append new images and labels to the existing data
    images = np.concatenate([existing_images, np.array(new_images)])
    labels = np.concatenate([existing_labels, np.array(new_labels)])

# Save images and labels together in a tuple
data = (images, labels)

# Dump the tuple containing both images and labels into the pickle file
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Print the shape of the loaded data
print("Shape of images array:", images.shape)
print("Shape of labels array:", labels.shape)

# Load the pickle file
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Extract images from the loaded data
loaded_images, _ = loaded_data

# Get the number of images
num_images = loaded_images.shape[0]

print("Number of images in the pickle file:", num_images)