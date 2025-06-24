# Necessary libraries for processing : 
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

# Libraries for training and validation
import ultralytics
from ultralytics import YOLO
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader

# Libraries to extract audio
from moviepy import *

# Unzip the zip file best.zip 
file = zipfile.ZipFile("./best.pt")
file.extractall()
file.close()

# Loading and changing the resnet model final layer as per our requirement
resnet = models.resnet50(pretrained = True)
num_classes = 3
resnet.fc = nn.Linear(resnet.fc.in_features,num_classes)

# Extracting the video frames from the downloaded video
direc = os.path.join('./Messi_Ronaldo.mp4')
video_pth = os.path.join('video_frames')
#video_pth = os.makedirs("video_frames")
cam = cv.VideoCapture(direc)
count = 0

while cam.isOpened():
    ret,frame = cam.read()
    cv.imwrite(os.path.join(video_pth,f'{count}.jpg'),frame)
    count += 1

cam.release()
cv.destroyAllWindows()

# Extracting and saving the audio of the video file 
def video_audio(vid_pth,output_pth):
    video = VideoFileClip(vid_pth)
    audio = video.audio
    audio.write_audiofile(output_pth)

video_pth = os.path.join('Messi_Ronaldo.mp4')
output_pth = 'short_audio.mp3'
video_audio(video_pth,output_pth)

# Creating labels for the Images
num_classes = 3
img_per_class = 20
labels = []
for cls in range(num_classes):
    labels.extend([cls]*img_per_class)

# Dataset path
img_pth = []
for pth in os.listdir('./Player Dataset'):
    im_pth = os.path.join('./Player Dataset',pth)
    for p in os.listdir(im_pth):
        img = os.path.join(im_pth,p)
        img_pth.append(img)

# Shuffling the path and labels together to maintain the mapped values intact
combined = list(zip(img_pth,labels))
random.shuffle(combined)
img_pth[:],labels[:] = zip(*combined)

# Dataset creation 
class PlayerDataset(Dataset):
    def __init__(self,img_pth,labels,transforms = None):
        self.img_pth = img_pth
        self.labels = labels
        self.transforms = transforms
    def __len__(self):
        return len(self.img_pth)
    def __getitem__(self,idx):
        img = Image.open(self.img_pth[idx]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[idx]
        return img,label

# Data Augmentation techniques for train and test data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Loading data
dataset = PlayerDataset(img_pth,labels,transforms = train_transform)
loader = DataLoader(dataset,batch_size = 32,shuffle = True)

# Mapping class names to the integer values
res_class = {0:'Messi',2:'Ronaldo',1:'No Class'}

# Finetuning the actual resnet model
epochs = 16
resnet.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr = 0.001,weight_decay = 1e-3)

for epoch in range(epochs) :
  running_loss = 0
  for img,label in loader:
    img, label = img,label
    output = resnet(img)
    loss = criterion(output,label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}')

  torch.save(resnet.state_dict(),'Resnet_Player.pth')

#Test Case : 
path = os.path.join('/home/user/ronaldo.jpeg')
img = Image.open(path).convert('RGB')
img = test_transform(img)
img = img.unsqueeze(0)
result = resnet(img)
idx = result.argmax()
print(res_class[int(idx)])