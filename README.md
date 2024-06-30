# MNIST-dataset
The MNIST dataset is a classic introductory dataset for deep learning, consisting of 60,000 training images and 10,000 test images of handwritten digits (0-9). We'll use a neural network to classify these digits.

Here’s a step-by-step guide to get started with the MNIST dataset using TensorFlow and Keras:

Step 1: Setup
First, ensure you have TensorFlow installed. You can install it using pip:

pip install tensorflow
Step 2: Import Libraries and Load Dataset
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print the shape of the data
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
Step 3: Preprocess the Data
# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
Step 4: Build the Model
model = Sequential([
    Flatten(input_shape=(28, 28)), # Flatten the input image
    Dense(128, activation='relu'), # First hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),  # Second hidden layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit) and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
Step 5: Train the Model
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
Step 6: Evaluate the Model
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
Step 7: Make Predictions
# Make predictions on the test data
predictions = model.predict(X_test)

# Show some sample predictions
for i in range(5):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"True label: {y_test[i].argmax()}, Predicted: {predictions[i].argmax()}")
    plt.show()
