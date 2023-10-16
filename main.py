import cv2
import numpy as np
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

    

def overlay_black_object(image, coin_location):    
    x, y, radius = coin_location
    radius = 50
    overlay_color = (0, 0, 0)  # Black color

    # Create a black rectangle to overlay over the coin
    image_overlayed = image.copy()
    
    start_x = max(0, x - radius)
    start_y = max(0, y - radius)
    end_x = min(image.shape[1], x + radius)
    end_y = min(image.shape[0], y + radius)

    image_overlayed[start_y:end_y, start_x:end_x] = np.clip(image_overlayed[start_y:end_y, start_x:end_x] - overlay_color, 0, 255)

    return image_overlayed



def detect_coins(image, waved):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    color = (0, 255, 0)
    thickness = 3

    if waved == True:
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

# Load pre-trained model for coin classification
# Replace this with your actual coin classification model and logic

# Initialize webcam
cap = cv2.VideoCapture(0)
waved = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect coins in the current frame
    coins_detected = detect_coins(frame, waved)

    # Detect hand wave
    if waved == False:
        hand_wave_detected = detect_hand_wave(frame)

    if hand_wave_detected:        
        waved = True    

    # Display the result
    cv2.imshow('Coin Detection', coins_detected)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
