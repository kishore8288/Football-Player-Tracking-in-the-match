# Necessary libraries for processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import cv2 as cv
import tensorflow as tf
import torch
import pickle
import zipfile
import os
import random
from PIL import Image

#Libraries for Inference 
import ultralytics
from ultralytics import YOLO
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader

# Importing yolo model for object detection : 
yolo = YOLO('yolov8n.pt')

# Loading weights that have been stored after traning the resnet model
resnet = models.resnet50(pretrained = True)
num_classes = 3
resnet.fc = nn.Linear(resnet.fc.in_features,num_classes)
resnet.load_state_dict(torch.load('Resnet_Player.pth'))

# Resnet and yolo classes 
classes = yolo.names
res_class = {0:'Messi',2:'Ronaldo',1:'No Class'}

# Making a list of frame names with proper order for future stitching these frames into a video
frame_dir = 'video_frames'
frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(os.path.splitext(x)[0]))
vid_dir = 'det_vid_frames'

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

""" Actual code to detect persons in an image and pass these persons to resnet model
    Model tries to map the person as the classified classes and respective bounding boxes
    are retrieved and the person with these labels get a bbox. These created new image frame
    is saved into another directory in the same order as extracted above. 
"""
count = 0
for frame in frame_files:
    img = cv.imread(os.path.join(frame_dir,frame))
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result = yolo(img)
    for r in result:
        for box in r.boxes:
            ind = int(box.cls)
            label = classes[ind]
            if label == 'person':
                b = box.xyxy[0]
                x1,y1,x2,y2 = map(int,b)
                c = box.data[:,4]
                if c >= 0.5:
                    im = img[y1:y2,x1:x2]
                    im = Image.fromarray(im)
                    im = test_transform(im).unsqueeze(0)
                    res = resnet(im)
                    out = res.argmax()
                    text = res_class[int(out)]
                    if text == 'Ronaldo':
                        cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),thickness = 2)
                        (text_w,text_h),baseline = cv.getTextSize(text,cv.FONT_HERSHEY_SIMPLEX,0.5,1)
                        cv.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1),(255,0,0), -1)
                        cv.putText(img, text, (x1 + 2, y1 - 4), cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), thickness=1, lineType=cv.LINE_AA)
                    elif text == 'Messi':
                        cv.rectangle(img,(x1,y1),(x2,y2),(0,255,255),thickness = 2)
                        (text_w,text_h),baseline = cv.getTextSize(text,cv.FONT_HERSHEY_SIMPLEX,0.5,1)
                        cv.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1),(0,255,255), -1)
                        cv.putText(img, text, (x1 + 2, y1 - 4), cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), thickness=1, lineType=cv.LINE_AA)
    cv.imwrite(os.path.join(vid_dir,f'{count}.jpg'),cv.cvtColor(img,cv.COLOR_RGB2BGR))
    count += 1
    print(f'frame_{count}.jpg saved')

# These new frames are reshaped to a new video and the audio of the previous video is added to it
frame_files = sorted(os.listdir(vid_dir), key=lambda x: int(os.path.splitext(x)[0]))
height, width, _ = cv.imread(os.path.join(vid_dir, frame_files[0])).shape

out = cv.VideoWriter("annotated_video.mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for frame in frame_files:
    img = cv.imread(os.path.join(vid_dir, frame))
    out.write(img)

out.release()