from ultralytics import YOLO
import cvzone
import math

model = YOLO("C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/weights/poker_best.pt")

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
    colors = [(255, 0, 0), (0, 255, 0)]
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
#         approx = cv2.approxPolyD P(contour, epsilon, True)

#         # Check if the polygon has 4 vertices (a quadrilateral)
#         if len(approx) == 4:
#             # Check if the aspect ratio is close to a poker card's aspect ratio
#             x, y, w, h = cv2.boundingRect(approx)
#             aspect_ratio = float(w) / h
#             if 2.3 <= aspect_ratio <= 2.7:
#                 card_rectangles.append(approx)

#     return card_rectangles

