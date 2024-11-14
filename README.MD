# Usage

## With pretrained weights

Modify <model_path> in the poker_detect.py

In Bash:
``python code/main.py``

## Without pretrained weights

Bash:
``pip install ultralytics``

``yolo task=detect mode=train model=yolov8n.pt data=<pokercard_dataset_path> epochs=50 imgsz=640``
