#!/usr/bin/env python
# coding: utf-8

# # Dice YOLOv8 Train and Predict
import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from PIL import Image
from IPython import get_ipython
import ultralytics
from ultralytics import YOLO
ultralytics.checks()

train_path='datasets/train/'
valid_path='datasets/valid/'
test_path='datasets/test/'


# # Data Preparation

# In[ ]:


ano_paths=[]
for dirname, _, filenames in os.walk('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Annotations'):
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

train_i = N[:train_size]
valid_i = N[train_size:train_size+valid_size]
test_i = N[train_size+valid_size:]

#print(train_i)
#print(valid_i)
#print(test_i)


# In[ ]:


for i in train_i:
    ano_path=ano_paths[i]
    img_path=os.path.join('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Images',
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    try:
        get_ipython().system('cp {ano_path} {train_path}')
        get_ipython().system('cp {img_path} {train_path}')
    except:
        continue
print(len(os.listdir(train_path)))


# In[ ]:


for i in valid_i:
    ano_path=ano_paths[i]
    img_path=os.path.join('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Images',
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    try:
        get_ipython().system('cp {ano_path} {valid_path}')
        get_ipython().system('cp {img_path} {valid_path}')
    except:
        continue
print(len(os.listdir(valid_path)))


# In[ ]:


for i in test_i:
    ano_path=ano_paths[i]
    img_path=os.path.join('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Images',
                          ano_path.split('/')[-1][0:-4]+'.jpg')
    try:
        get_ipython().system('cp {ano_path} {test_path}')
        get_ipython().system('cp {img_path} {test_path}')
    except:
        continue
print(len(os.listdir(test_path)))      


# # Create yaml file 

import yaml

data_yaml = dict(
    train ='train',
    val ='valid',
    test='test',
    nc =6,
    names =list('123456')
)

with open('data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)
    
with open('data.yaml', 'r') as file:
    data = file.read()
import yaml

with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

names = data['names']
M = list(range(len(names)))
class_map = dict(zip(M, names))

# # Train

import subprocess

model = YOLO("yolov8x.pt") 

subprocess.run(['yolo', 'task=detect', 'mode=train', 'model=yolov8x.pt', 'data=data.yaml', 'epochs=12', 'imgsz=480'])

# # Result of Training

paths2=[]
for dirname, _, filenames in os.walk('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/train'):
    for filename in filenames:
        if filename[-4:]=='.jpg':
            paths2+=[(os.path.join(dirname, filename))]
paths2=sorted(paths2)


for path in paths2:
    image = Image.open(path)
    image=np.array(image)
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.show()


# # Predict


best_path0='runs/detect/train/weights/best.pt'
source0='datasets/test'



ppaths=[]
for dirname, _, filenames in os.walk(source0):
    for filename in filenames:
        if filename[-4:]=='.jpg':
            ppaths+=[(os.path.join(dirname, filename))]
ppaths=sorted(ppaths)
print(ppaths[0])
print(len(ppaths))



model2 = YOLO(best_path0)

# predict
source0 = 'datasets/test'
conf = 0.2
results = model2.predict(source0, conf=conf)
print(len(results))



print((results[0].boxes.data))



PBOX=pd.DataFrame(columns=range(6))
for i in range(len(results)):
    arri=pd.DataFrame(results[i].boxes.data.cpu().numpy()).astype(float)
    path=ppaths[i]
    file=path.split('/')[-1]
    arri=arri.assign(file=file)
    arri=arri.assign(i=i)
    PBOX=pd.concat([PBOX,arri],axis=0)
PBOX.columns=['x','y','x2','y2','confidence','class','file','i']
print(PBOX)

PBOX['class']=PBOX['class'].apply(lambda x: class_map[int(x)])
PBOX=PBOX.reset_index(drop=True)
print(PBOX)
print(PBOX['class'].value_counts())


def draw_box2(n0):
    
    ipath=ppaths[n0]
    image=cv2.imread(ipath)
    H,W=image.shape[0],image.shape[1]
    file=ipath.split('/')[-1]  
    
    if PBOX[PBOX['file']==file] is not None:
        box=PBOX[PBOX['file']==file]
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


def create_animation(ims):
    fig=plt.figure(figsize=(12,8))
    im=plt.imshow(cv2.cvtColor(ims[0],cv2.COLOR_BGR2RGB))
    text = plt.text(0.05, 0.05, f'Slide {0}', transform=fig.transFigure, fontsize=14, color='blue')
    plt.axis('off')
    plt.close()

    def animate_func(i):
        im.set_array(cv2.cvtColor(ims[i],cv2.COLOR_BGR2RGB))
        text.set_text(f'Slide {i}')        
        return [im]    
    
    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000)


images2=[]
for i in tqdm(range(len(ppaths))):
    images2+=[draw_box2(i)]


create_animation(images2)




import os
import random
import yaml
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from yolov8 import YOLO

# Define paths for train, validation and test data
train_path = 'datasets/train'
valid_path = 'datasets/valid'
test_path = 'datasets/test'

# Create directories for train, validation and test data
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get all annotation file paths
ano_paths = []
for dirname, _, filenames in os.walk('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Annotations'):
    for filename in filenames:
        if filename[-4:] == '.txt':
            ano_paths.append(os.path.join(dirname, filename))

# Shuffle the annotation file paths
n = len(ano_paths)
N = list(range(n))
random.shuffle(N)

# Split the annotation file paths into train, validation and test sets
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

train_size = int(train_ratio * n)
valid_size = int(valid_ratio * n)

train_i = N[:train_size]
valid_i = N[train_size:train_size + valid_size]
test_i = N[train_size + valid_size:]

# Copy the annotation and image files to the train, validation and test directories
for i in train_i:
    ano_path = ano_paths[i]
    img_path = os.path.join('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Images',
                            ano_path.split('/')[-1][0:-4] + '.jpg')
    try:
        os.system(f'cp {ano_path} {train_path}')
        os.system(f'cp {img_path} {train_path}')
    except:
        continue
print(len(os.listdir(train_path)))

for i in valid_i:
    ano_path = ano_paths[i]
    img_path = os.path.join('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Images',
                            ano_path.split('/')[-1][0:-4] + '.jpg')
    try:
        os.system(f'cp {ano_path} {valid_path}')
        os.system(f'cp {img_path} {valid_path}')
    except:
        continue
print(len(os.listdir(valid_path)))

for i in test_i:
    ano_path = ano_paths[i]
    img_path = os.path.join('C:/Users/kabaz_kwuenmy/Documents/GitHub/cv-magic-trick/d6-dice-dataset/Images',
                            ano_path.split('/')[-1][0:-4] + '.jpg')
    try:
        os.system(f'cp {ano_path} {test_path}')
        os.system(f'cp {img_path} {test_path}')
    except:
        continue
print(len(os.listdir(test_path)))

# Create data.yaml file
data_yaml = dict(
    train='train',
    val='valid',
    test='test',
    nc=6,
    names=list('123456')
)

with open('data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)

with open('data.yaml', 'r') as f:
    print(f.read())

# Define class map
names = list('123456')
M = list(range(len(names)))
class_map = dict(zip(M, names))

# Train the model
model = YOLO("yolov8x.pt")
import subprocess

subprocess.run(['yolo', 'task=detect', 'mode=train', 'model=yolov8x.pt', 'data=data.yaml', 'epochs=12', 'imgsz=480'])

# Predict using the trained model
best_path0 = 'runs/detect/train/weights/best.pt'
source0 = 'datasets/test'

model2 = YOLO(best_path0)
subprocess.run(['yolo', 'task=detect', 'mode=predict', f'model={best_path0}', 'conf=0.2', f'source={source0}'])

# Process the prediction results
results = model2.predict(source0, conf=0.2)
print(len(results))

PBOX = pd.DataFrame(columns=range(6))
ppaths = []
for dirname, _, filenames in os.walk(source0):
    for filename in filenames:
        if filename[-4:] == '.jpg':
            ppaths.append(os.path.join(dirname, filename))
ppaths = sorted(ppaths)

for i in range(len(results)):
    arri = pd.DataFrame(results[i].boxes.data.cpu().numpy()).astype(float)
    path = ppaths[i]
    file = path.split('/')[-1]
    arri = arri.assign(file=file)
    arri = arri.assign(i=i)
    PBOX = pd.concat([PBOX, arri], axis=0)
PBOX.columns = ['x', 'y', 'x2', 'y2', 'confidence', 'class', 'file', 'i']
display(PBOX)

PBOX['class'] = PBOX['class'].apply(lambda x: class_map[int(x)])
PBOX = PBOX.reset_index(drop=True)
display(PBOX)
display(PBOX['class'].value_counts())

# Draw bounding boxes on the images
def draw_box2(n0):
    ipath = ppaths[n0]
    image = cv2.imread(ipath)
    H, W = image.shape[0], image.shape[1]
    file = ipath.split('/')[-1]

    if PBOX[PBOX['file'] == file] is not None:
        box = PBOX[PBOX['file'] == file]
        box = box.reset_index(drop=True)
        for i in range(len(box)):
            label = box.loc[i, 'class']
            x = int(box.loc[i, 'x'])
            y = int(box.loc[i, 'y'])
            x2 = int(box.loc[i, 'x2'])
            y2 = int(box.loc[i, 'y2'])
            cv2.putText(image, f'{label}', (x, int(y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 3)
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 3)

    return image

# Create animation of the images with bounding boxes
images2 = []
for i in tqdm(range(len(ppaths))):
    images2.append(draw_box2(i))

create_animation(images2)
