import cv2
import os
import numpy as np
import csv

# Directories containing the images
directories = ["./Data/A", "./Data/B"]

# Initialize arrays to store pixel values for x and y dimensions
x_values = []
y_values = []
targets = []

# Iterate through each directory
for idx, directory in enumerate(directories):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        continue

    # Iterate through each image file in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        filepath = os.path.join(directory, filename)

        # Read the image using OpenCV
        img = cv2.imread(filepath)

        # Check if the image was loaded successfully
        if img is not None:
            # Extract dimensions of the image
            height, width, _ = img.shape

            # Assign the target value for this image
            target = idx

            # Iterate through each pixel in the image
            for y in range(height):
                for x in range(width):
                    # Get the pixel value at the current position (x, y)
                    pixel_value = img[y, x]

                    # Store the pixel values and target value in the arrays
                    x_values.append(pixel_value[0])  # Blue channel
                    y_values.append(pixel_value[1])  # Green channel
                    targets.append(target)  # Assign the target value for this pixel
        else:
            print(f"Failed to load image: {filepath}")

# Convert the lists to numpy arrays
x_values = np.array(x_values)
y_values = np.array(y_values)
targets = np.array(targets)

# Now, x_values, y_values, and targets arrays contain the pixel values and target variables for all images
# You can use these arrays for further analysis or processing

# Combine x_values, y_values, and targets into a single array
data = np.column_stack((x_values, y_values, targets))

# Define the path to save the CSV file
csv_file_path = "image_data.csv"

# Save the data to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['x_value', 'y_value', 'target'])
    # Write the data rows
    writer.writerows(data)

print("Data saved to", csv_file_path)
