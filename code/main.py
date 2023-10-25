import cv2

from hand_wave_detect import detect_hand_wave
from coin_detect import detect_coins
from poker_detect import poker_detection
from dice_detect import dice_detection

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
waved = False
visible = True

while True:
    # Set the parameters to detect hand wave
    hand_wave_detected = False
    ret, frame = cap.read()
    if not ret:
        break

    ### Hand Wave Detection ###
    hand_wave_detected = detect_hand_wave(frame)

    if hand_wave_detected:
        waved = not waved
        visible = waved

    ### Coin Detection ###
    coins_detected = detect_coins(frame, visible)

    ### Poker Detection ###
    poker_detected = poker_detection(frame)

    ### Dice Detection ###
    dice_detected = dice_detection(frame)

    # Display the result
    cv2.imshow('Coin/Card Detection', coins_detected)
    cv2.imshow('Coin/Card Detection', poker_detected)
    cv2.imshow('Coin/Card Detection', dice_detected)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
