import hashlib
import json
from typing import List, Tuple
import torch
from os import listdir
import clip
import os
from PIL import Image
import numpy as np
import torch
import cv2
import pickle
from os.path import exists


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(
    app,
    resources={
        r"/classify": {"origins": "*"},
        r"/network-layout": {"origins": "*"},
        r"/trace": {"origins": "*"},
        r"/trace-without-weight": {"origins": "*"},
        r"/trace-only-weight": {"origins": "*"},
    },
)
VIDEO_DIR = "./video-data"
FEATURE_DIR = "./video-features"
FRAMES_DIR = "./video-frames"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# model.cuda().eval()


def get_video_metadata(filename: str, index: List[dict]) -> Tuple[dict, int]:
    for i, meta in enumerate(index):
        if meta["name"] == filename:
            return meta, i
    return None, -1


@app.route("/probability", methods=["GET"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def probability():
    phrase = request.args.get('phrase')
    files = json.loads(request.args.get('files'))

    with open('./index.json', "r") as f:
        index = json.load(f)

    features = []
    video_indices = []
    frame_order = []
    for file in files:
        meta, i = get_video_metadata(file, index)
        if meta is None:
            return f"file not found: {file}", 500
        with open(os.path.join(FEATURE_DIR, f'{file}.features.pickle'), "rb") as f:
            feature = pickle.load(f)
            assert feature.shape[0] == meta["frames"]
            features.append(feature)
            video_indices.append(np.array([i] * int(meta["frames"])))
            frame_order.append(np.array([*range(int(meta["frames"]))]))
    
    np_features = np.concatenate(features)
    np_video_indices = np.concatenate(video_indices)
    np_frame_order = np.concatenate(frame_order)

    np_features /= torch.from_numpy(np_features).norm(dim=-1, keepdim=True)

    text_tokens = clip.tokenize([phrase]) #.cuda()

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ np_features.cpu().numpy().T
    return str(similarity.tolist())


@app.route("/find", methods=["GET"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def find():
    phrase = request.args.get('phrase')
    files = json.loads(request.args.get('files'))
    sections = int(request.args.get('sections'))

    with open('./index.json', "r") as f:
        index = json.load(f)

    features = []
    video_indices = []
    frame_order = []
    for file in files:
        meta, i = get_video_metadata(file, index)
        if meta is None:
            return f"file not found: {file}", 500
        with open(os.path.join(FEATURE_DIR, f'{file}.features.pickle'), "rb") as f:
            feature = pickle.load(f)
            assert feature.shape[0] == meta["frames"]
            features.append(feature)
            video_indices.append(np.array([i] * int(meta["frames"])))
            frame_order.append(np.array([*range(int(meta["frames"]))]))
    
    np_features = np.concatenate(features)
    np_video_indices = np.concatenate(video_indices)
    np_frame_order = np.concatenate(frame_order)

    np_features /= torch.from_numpy(np_features).norm(dim=-1, keepdim=True)

    text_tokens = clip.tokenize([phrase]) #.cuda()

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (text_features.cpu().numpy() @ np_features.cpu().numpy().T)[0]
    similarity_order = similarity.argsort(similarity)

    ordered_video_indices = np_video_indices[similarity_order]
    ordered_frame_order = np_frame_order[similarity_order]

    for vi, fo in zip(ordered_video_indices, ordered_frame_order):
        pass



if __name__ == "__main__":
    app.run(port=5432)