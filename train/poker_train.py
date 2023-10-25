### Download dataset from https://universe.roboflow.com/roboflow-100/poker-cards-cxcvz
!yolo task=detect mode=train model=yolov8n.pt data='<dataset_path>' epochs=50 imgsz=640