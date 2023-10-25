import cv2
import numpy as np
from ultralytics import YOLO

def detect_coins(image, visible):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    color = (0, 255, 0)
    thickness = 3

    if visible == False:
        color = (0, 0, 0)
        thickness = -1
    
    # Use Hough Circle Transform to detect circles (coins)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=40
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(image, center, radius, color, thickness)

    return image