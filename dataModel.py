import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Extract images and labels from the loaded data
images, labels = loaded_data

# Flatten the images and convert them to a 2D array
# Note: Depending on your model architecture, you might need to reshape the images differently
num_samples, height, width, channels = images.shape
images = images.reshape(num_samples, -1)

# Convert labels into numerical format (if they are not already)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess the data as needed (e.g., normalize pixel values)

# Define and train your model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = accuracy_score(y_train, model.predict(X_train))
val_accuracy = accuracy_score(y_val, model.predict(X_val))

print("Training Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
