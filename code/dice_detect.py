from ultralytics import YOLO
import cvzone
import math

dice_model = YOLO("C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/weights/dice_best.pt")

classNames = ['1', '2', '3', '4', '5', '6']

def dice_detection(img):
    results = dice_model(img, stream=True)
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

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    return img