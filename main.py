import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Distance calibration data and coefficients
raw_distances = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
distances_cm = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
A, B, C = np.polyfit(raw_distances, distances_cm, 2)  # y = Ax^2 + Bx + C

# Loop for video feed and hand detection
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)  # Unpack the result here

    if hands:
        lmList = hands[0]['lmList']
        bbox = hands[0]['bbox']
        x1, y1 = lmList[5][:2]  # Extract only x and y coordinates
        x2, y2 = lmList[17][:2]

        # Calculate the Euclidean distance
        distance_pixels = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        distance_cm = A * distance_pixels ** 2 + B * distance_pixels + C

        # Draw bounding box and display distance
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cvzone.putTextRect(img, f'{int(distance_cm)} cm', (x + 5, y - 10))

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
