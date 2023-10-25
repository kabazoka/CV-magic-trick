import cv2
import mediapipe as mp

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