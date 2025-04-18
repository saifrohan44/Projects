import cv2
import os

# Create directories for saving images
for i in range(1, 4):
    os.makedirs(f'person_{i}', exist_ok=True)

# Initialize the webcam
cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture Images")

# Loop through for each person
for person_id in range(1, 4):
    print(f"Collecting images for person {person_id}...")
    count = 0
    while count < 100:  # Capture 100 images per person
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1)
        if key % 256 == 32:  # Press SPACE to capture
            img_name = f"person_{person_id}/image_{count}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Image {count} saved for person {person_id}")
            count += 1
        elif key % 256 == 27:  # Press ESC to exit
            break

cam.release()
cv2.destroyAllWindows()