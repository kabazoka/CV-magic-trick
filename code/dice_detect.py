from ultralytics import YOLO
import cvzone
import math
import cv2

dice_model = YOLO("weights/poker_best.pt")

classNames = ['1', '2', '3', '4', '5', '6']

def dice_detection(img):
    results = dice_model(img, stream=True)
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # if x2 > 320 and y2 > 240:
            #     pos = 1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            if conf > 0.5:
                hand.append(classNames[cls])

    return img

def draw_box2(frame):
    results = dice_model(frame, stream=True)
    ipath=frame
    image=frame
    H,W=image.shape[0],image.shape[1]

    for r in results:
        box = r.box        
        box=box.reset_index(drop=True)
        #display(box)
        for i in range(len(box)):
            label=box.loc[i,'class']
            x=int(box.loc[i,'x'])
            y=int(box.loc[i,'y'])
            x2=int(box.loc[i,'x2']) 
            y2=int(box.loc[i,'y2'])
            #print(label,x,y,x2,y2)
            cv2.putText(image, f'{label}', (x,int(y-4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0,(0,255,0),3)
            cv2.rectangle(image,(x,y),(x2,y2),(0,255,0),3)
    
    return image