# yolov8n.pt file 
from google.colab import drive
drive.mount('/content/drive') # here I simply stored my file in google drive
!pip install ultralytics
from  ultralytics import YOLO
!yolo task=detect mode=train model=yolov8n.pt data='Your dataset path in google drive' epochs=50 imgsz=640
# Results will be saved to runs/detect/predict in drive
# you need to download that pre-trained prediction file to work with.