import cv2
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO
import cvzone
import math


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model = YOLO("best.pt")
classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

def poker_detection(img):
    results = model(img, stream=True)
    hand = []
    colors = [(255, 0, 0), (0, 255, 0)] #BGR
    pos = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            if x2 > 320 and y2 > 240:
                pos = 1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorT=colors[pos])

            if conf > 0.5:
                hand.append(classNames[cls])
    return img


def detect_hand_wave(frame):
    # Initialize MediaPipe Hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Convert the image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(frame_rgb)                    

        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Get landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get the y-coordinate of the middle finger's tip
            middle_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            
            # Check if the middle finger's tip is above a threshold (hand wave motion)
            if middle_finger_tip_y < 0.5:                
                return True
            else:
                return False
        else:
            return False
        

# def identify_card_rectangles(frame):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply GaussianBlur to reduce noise and improve contour detection
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Use Canny edge detection to detect edges
#     edges = cv2.Canny(blurred, 50, 150)

#     # Find contours in the edge-detected image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     card_rectangles = []
#     for contour in contours:
#         # Approximate the contour to a polygon
#         epsilon = 0.02 * cv2.arcLength(contour, True)  # Adjust epsilon value as needed
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Check if the polygon has 4 vertices (a quadrilateral)
#         if len(approx) == 4:
#             # Check if the aspect ratio is close to a poker card's aspect ratio
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
#             if 2.3 <= aspect_ratio <= 2.7:
#                 card_rectangles.append(approx)

#     return card_rectangles


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

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
waved = False
visible = True

while True:
    # Hand wave detection
    hand_wave_detected = False
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hand wave
    hand_wave_detected = detect_hand_wave(frame)

    if hand_wave_detected:
        waved = not waved
        visible = waved

    # Detect coins in the current frame
    coins_detected = detect_coins(frame, visible)

    ### Poker Detection ###
    poker_detected = poker_detection(frame)

    # Display the result
    cv2.imshow('Coin/Card Detection', coins_detected)
    cv2.imshow('Coin/Card Detection', poker_detected)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
