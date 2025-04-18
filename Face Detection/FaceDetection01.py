import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('face_recognition_cnn_model02.h5')

# Class labels
people = ['moynul', 'riya', 'rohan']  # Updated list of people

# Haar Cascade for face detection
haarCascade = cv.CascadeClassifier('harr_face.xml')

# Start webcam feed
cap = cv.VideoCapture(0)  # Use 0 for the default webcam. Change to 1 or other index if needed.

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from the webcam. Exiting...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces in the frame
    faces_rect = haarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        # Extract the region of interest (face)
        face_roi = gray[y:y + h, x:x + w]
        resized_face = cv.resize(face_roi, (600, 400))  # Resize to the CNN input size
        normalized_face = resized_face / 255.0  # Normalize pixel values
        reshaped_face = normalized_face.reshape(1, 600, 400, 1)  # Add batch and channel dimensions

        # Predict the label using the CNN model
        predictions = model.predict(reshaped_face)
        confidence = np.max(predictions) * 100  # Get the confidence percentage
        if confidence < 50:
            label_text = f'Unknown Person ({confidence:.2f}%)'
        else:
            label = np.argmax(predictions)  # Get the index of the highest probability
            label_text = f'{people[label]} ({confidence:.2f}%)'

        # Display label and confidence on the video feed
        cv.putText(frame, label_text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the real-time video feed
    cv.imshow("Webcam - Face Recognition", frame)

    # Exit the webcam feed when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
