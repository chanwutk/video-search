from dataclasses import dataclass, asdict
import json
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
import clip
from os.path import join
import pickle


VIDEO_DIR = "video-data"
FEATURE_DIR = "video-features"


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# model.cuda().eval()


@dataclass
class VideoMeta:
    name: str
    fps: float
    width: int
    height: int
    frames: int


@dataclass
class Video:
    __data_dir: str
    __meta: VideoMeta
    __content: np.ndarray | None
    __features: torch.Tensor | None

    @property
    def meta(iself):
        return iself.__meta
    
    @property
    def content(self):
        if self.__content is None:
            with open(_content_path(self.__data_dir, self.meta.name), "rb") as f:
                self.__content = pickle.load(f)
        return self.__content
    
    @property
    def features(self):
        if self.__features is None:
            with open(_features_path(self.__data_dir, self.meta.name), "rb") as f:
                self.__features = torch.from_numpy(pickle.load(f))
        return self.__features
    

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)


def _content_path(data_path: str, name: str) -> str:
    return join(data_path, VIDEO_DIR, f"{name}.content.pickle")


def _features_path(data_path: str, name: str) -> str:
    return join(data_path, FEATURE_DIR, f"{name}.features.pickle")


class VideoStore:
    data_path: str
    videos: List[Video]
    names: Set[str]
    def __init__(self, data_path: str):
        self.data_path = data_path
        with open(join(self.data_path, "index.json"), "r") as f:
            index = json.load(f)
        
        self.videos = [Video(VideoMeta(**i), None, None) for i in index]
        self.names = set([i["name"] for i in index])
    
    def video(self, meta: VideoMeta, content: np.ndarray | None, features: torch.Tensor | None) -> Video:
        return Video(self.data_path, meta, content, features)

    def add(self, name: str, fps: float, content: np.ndarray) -> "VideoStore":
        if name in self.names:
            raise Exception(f"Video with name '{name}' already exists")
        self.names.add(name)
        frames, width, height, channel = content.shape
        assert channel == 3

        image_input = torch.tensor(content) #.cuda()
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
        
        self.videos.append(self.video(
            VideoMeta(name, fps, width, height, frames),
            content,
            image_features
        ))

        with open(_content_path(self.data_path, name), "wb") as f:
            pickle.dump(content, f)
        with open(_features_path(self.data_path, name), "wb") as f:
            pickle.dump(image_features.numpy(), f)
        with open(join(self.data_path, "index.json"), "w") as f:
            json.dump([asdict(v) for v, _, _ in self.videos])
        
        return self

    def list_videos(self) -> List[VideoMeta]:
        return [v.meta for v in self.videos]
    
    def get_videos(self, names: List[str]) -> List[Tuple[VideoMeta, np.ndarray]]:
        _names = set(names)
        return [
            (v.meta, v.content)
            for v in self.videos
            if v.meta.name in _names
        ]

    def probability(
        self,
        phrase: str,
        videos: List[str]
    ) -> Dict[str, List[float]]:
        for name in videos:
            if name not in self.names:
                raise Exception(f"Video with name '{name}' does not exist")

        features = []
        video_indices = []
        frame_order = []
        for i, video in enumerate(self.videos):
            v, _, f = video
            if v.names not in videos: continue


        for file in videos:
            meta, i = get_video_metadata(file, index)
            if meta is None:
                return f"file not found: {file}", 500
            with open(os.path.join(FEATURE_DIR, f'{file}.features.pickle'), "rb") as f:
                feature = pickle.load(f)
                assert feature.shape[0] == meta["frames"]
                features.append(feature)
                video_indices.append(np.array([i] * int(meta["frames"])))
                frame_order.append(np.array([*range(int(meta["frames"]))]))
        pass

    def find(
        self,
        phrase: str,
        videos: List[str],
        num_slice: int,
        threshold: float
    ) -> List[Tuple[str, Tuple[int, int]]]:
        pass
