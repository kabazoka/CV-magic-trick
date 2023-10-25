import os
import random
from shutil import copy
import ultralytics
from ultralytics import YOLO

#dataset splitting

train_path='C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/datasets/train'
valid_path='C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/datasets/valid'
test_path='C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/datasets/test'
img_folder_path='C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/src/d6-dice-dataset/Images'

ano_paths=[]
for dirname, _, filenames in os.walk('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/src/d6-dice-dataset/Annotations'):
    for filename in filenames:
        if filename[-4:]=='.txt':
            ano_paths+=[(os.path.join(dirname, filename))]
        
n=len(ano_paths) 
print(n)
N=list(range(n))
random.shuffle(N)

train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

train_size = int(train_ratio*n)
valid_size = int(valid_ratio*n)

train_num = N[:train_size]
valid_num = N[train_size:train_size+valid_size]
test_num = N[train_size+valid_size:]

#print(train_num)
#print(valid_num)
#print(test_num)


for i in train_num:
    ano_path=ano_paths[i].replace("\\", "/")
    img_path=os.path.join(img_folder_path,
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    img_path.replace("\\", "/")
    try:
        copy(ano_path, train_path+'/labels')
        copy(img_path, train_path+'/images')    
    except:
        continue

print(len(os.listdir(train_path)))

for i in valid_num:
    ano_path=ano_paths[i].replace("\\", "/")
    img_path=os.path.join(img_folder_path,
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    img_path.replace("\\", "/")
    try:
        copy(ano_path, valid_path+'/labels')
        copy(img_path, valid_path+'/images')    
    except:
        continue

print(len(os.listdir(valid_path)))

for i in test_num:
    ano_path=ano_paths[i].replace("\\", "/")
    img_path=os.path.join(img_folder_path,
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    img_path.replace("\\", "/")
    try:
        copy(ano_path, test_path+'/labels')
        copy(img_path, test_path+'/images')    
    except:
        continue

print(len(os.listdir(test_path)))

# yolov8n.pt train

# yolo task=detect mode=train model=yolov8n.pt data='datasets' epochs=50 imgsz=640

model = YOLO("yolov8x.pt") 
!yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=12 imgsz=480

# Results will be saved to runs/detect/predict in drive
# you need to download that pre-trained prediction file to work with.