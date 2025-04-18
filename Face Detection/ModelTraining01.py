import os
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Directory structure
DIR = r'data\train'
people = ['moynul', 'riya', 'rohan']  # Updated list of people

# Haar Cascade for face detection
haarCascade = cv.CascadeClassifier('harr_face.xml')

# Features and labels
features = []
labels = []

def create_training_data():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                # Resize face ROI to a fixed size for CNN input
                resized_face = cv.resize(faces_roi, (150, 150))
                features.append(resized_face)
                labels.append(label)

create_training_data()
print('Training data preparation done.')

# Convert to numpy arrays
features = np.array(features, dtype='float32') / 255.0  # Normalize pixel values
labels = np.array(labels)

# Reshape features to include channel dimension (CNN input format)
features = features.reshape(-1, 150, 150, 1)  # 1 channel for grayscale images

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=len(people))
y_val = to_categorical(y_val, num_classes=len(people))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(people), activation='softmax')  # Number of classes (3 for the updated dataset)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# Save the trained model
model.save('face_recognition_cnn_model.h5')
print("Model training complete and saved.")
