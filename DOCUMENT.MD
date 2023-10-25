# HW1 Magic Trick

## Tools

OpenCV, Mediapipe, YOLOv8

### Part 1: Coin

My code is able to detect the coins in the frame. And when a handwave is detected, the coin dis/appears. The disappear of the coin is simply implemented by covering a black circle to the position of the coins detected.

### Part 2: Poker card

My program is able to detect a poker card in the frame, and cover a poker card up on the position of the card detected. I trained a yolov8 model with the dataset.

### Part 3: Dice

I have trained a model with yolov8 with the same method as the poker card. But I’m having troubles with the real time detection on the dice. So, this function is not working for now.

## Why I choose AI

The reason why I choose to apply Deep Learning on the project is on my previous attempts(based on geometry detections with opencv), the detection of the poker and dice would have very much of confliction.

## Problems Encountered

For the Poker Card Detection: The problem I am having is the detection of the card is not suitable for white cards. Because the original requirement of the assignment did not described enough so I didn’t realize we have to detect the white card and apply tranformation on it.

For the DICE Detection: The model I trained is having troubles with real time detection.