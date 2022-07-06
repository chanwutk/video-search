import json
from typing import List
import torch
from os import listdir
import clip
import os
from PIL import Image
import numpy as np
import torch
import cv2
import pickle


VIDEO_DIR = "./video-data"
FEATURE_DIR = "./video-features"
FRAMES_DIR = "./video-frames"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# model.cuda().eval()


index = []

for filename in listdir(VIDEO_DIR):
    print(filename)
    images = []
    original_images: List[Image.Image] = []
    cam = cv2.VideoCapture(os.path.join(VIDEO_DIR, filename))
    width  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    index.append({
        "name": filename,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": int(frames),
    })

    while(True):
        ret, frame = cam.read()
        if not ret:
            break
    
        image = Image.fromarray(frame[:, :, ::-1], 'RGB')
        original_images.append(image)
        images.append(preprocess(image))


    image_input = torch.tensor(np.stack(images)) #.cuda()


    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    # image_features /= image_features.norm(dim=-1, keepdim=True)
    
    with open(os.path.join(FEATURE_DIR, f'{filename}.features.pickle'), 'wb') as handle:
        pickle.dump(image_features, handle)
    with open(os.path.join(FRAMES_DIR, f'{filename}.frames.pickle'), 'wb') as handle:
        pickle.dump(original_images, handle)

with open("./index.json", "w") as f:
    json.dump(index, f)