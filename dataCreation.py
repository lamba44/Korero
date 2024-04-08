import os
import numpy as np
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize

# Load and preprocess data
def load_data(data_dir):
    images = []
    labels = []
    classes = os.listdir(data_dir)
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):  # Check if the item is a directory
            print(f"Loading images from class directory: {class_dir}")
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if os.path.isfile(image_path):  # Check if the item is a file
                    print(f"Loading image: {image_path}")
                    image = imread(image_path)
                    image = resize(image, (64, 64))  # Resize images to 64x64 pixels
                    image = rgb2gray(image)  # Convert the image to grayscale
                    images.append(image.flatten())  # Flatten the image into a 1D array
                    labels.append(i)
                else:
                    print(f"Skipping non-file item: {image_path}")
        else:
            print(f"Skipping non-directory item: {class_dir}")
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

data_dir = './Data/A'
images, labels = load_data(data_dir)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=0.0001, solver='adam', random_state=42)
clf.fit(x_train, y_train)

# Predict and evaluate
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')
