# Football-Player-Tracking-in-the-match
[code](https://github.com/kishore8288/Football-Player-Tracking-in-the-match/blob/main/Football%20Player%20Prediction.ipynb)

## STEP-1: Install Dependencies
`Step to procede, setup and run the code:`
1. Download the [dataset](https://www.kaggle.com/datasets/kishore8824/messi-ronaldo-classification-dataset) and the video file from my repo.
2. Install **python** or **Jupyter notebook**
3. Installing Necessary libraries
   - Numpy : ```pip install numpy```
   - Matplotlib : ```pip install matplotlib```
   - cv2 : ```pip install opencv-python```
   - PIL : ```pip install pillow```
   - Tensorflow : ```pip install tensorflow```
   - Pytorch : ```pip install torch torchvision torchaudio```
   - Ultralytics : ```pip install ultralytics```
   - MoviePY : ```pip install moviepy```

> [!NOTE]
> Use the above commands only in the command prompt or terminal

## STEP-2 : Training ResNet
**Run ResnetTrain.py code in your terminal**\
`python ResnetTrain.py`/
or I will provide the resnet model weights [here](https://drive.google.com/file/d/1_iPCA7_PRhZK7xwhJXMLtTOO6QwSozXp/view?usp=sharing)\
- [x] #Just download it and go to step-3

## STEP-3 : Inference stage
**Run PlayerDetectionLocalizationVideo.py in you terminal as stated above**\

>[!NOTE]
> To get the audio of the original file to the extracted video file, use this command :
> ```ffmpeg -i extracted.mp4 -i original_video.mp4 -c:v copy -map 0:v:0 -map 1:a:0 -shortest extracted_AV.mp4```
