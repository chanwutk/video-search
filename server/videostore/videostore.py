from dataclasses import dataclass, asdict
import json
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import torch
import clip  # type: ignore
from os.path import join, isdir
from os import mkdir
import pickle


VIDEO_DIR = "video-data"
FEATURE_DIR = "video-features"


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # type: ignore


@dataclass
class VideoMeta:
    name: str
    fps: float
    width: int
    height: int
    frames: int


Array4D = np.ndarray[Tuple[int, int, int, int], np.dtype[np.float32]]
Array2D = np.ndarray[Tuple[int, int], np.dtype[np.float32]]


@dataclass
class Video:
    __data_dir: str
    __meta: VideoMeta
    __content: Array4D | None
    __features: Array2D | None

    @property
    def meta(iself):
        return iself.__meta

    @property
    def content(self):
        if self.__content is None:
            with open(_content_path(self.__data_dir, self.meta.name), "rb") as f:
                array = pickle.load(f)
                if not isinstance(array, np.ndarray):
                    raise Exception("loaded pickle should an np.ndarray")
                self.__content = array
        return self.__content

    @property
    def features(self):
        if self.__features is None:
            with open(_features_path(self.__data_dir, self.meta.name), "rb") as f:
                array = pickle.load(f)
                if not isinstance(array, np.ndarray):
                    raise Exception("loaded pickle should an np.ndarray")
                self.__features = array
        return self.__features


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    norm = features.norm(dim=-1, keepdim=True)  # type: ignore
    if not isinstance(norm, torch.Tensor):
        raise Exception()
    return features / norm


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
        index: List[Dict[str, Any]] = []
        if isdir(data_path):
            with open(join(self.data_path, "index.json"), "r") as f:
                index = json.load(f)
        else:
            mkdir(data_path)
            mkdir(join(data_path, FEATURE_DIR))
            mkdir(join(data_path, VIDEO_DIR))
            with open(join(self.data_path, "index.json"), "w") as f:
                f.write("[]")

        self.videos = [Video(data_path, VideoMeta(**i), None, None) for i in index]
        self.names = set([i["name"] for i in index])

    def video(
        self, meta: VideoMeta, content: Array4D | None, features: Array2D | None
    ) -> Video:
        return Video(self.data_path, meta, content, features)

    def add(self, name: str, fps: float, content: Array4D) -> "VideoStore":
        if name in self.names:
            raise Exception(f"Video with name '{name}' already exists")
        self.names.add(name)
        frames, channel, width, height = content.shape
        assert channel == 3, content.shape

        image_input = torch.tensor(content).to(device)
        with torch.no_grad():
            image_features: torch.Tensor = model.encode_image(image_input).float()  # type: ignore

        np_features: Array2D = image_features.numpy()
        self.videos.append(
            self.video(
                VideoMeta(name, fps, width, height, frames), content, np_features
            )
        )

        with open(_content_path(self.data_path, name), "wb") as f:
            pickle.dump(content, f)
        with open(_features_path(self.data_path, name), "wb") as f:
            pickle.dump(np_features, f)
        with open(join(self.data_path, "index.json"), "w") as f:
            json.dump([asdict(v.meta) for v in self.videos], f)

        return self

    def list_videos(self) -> List[VideoMeta]:
        return [v.meta for v in self.videos]

    def get_videos(self, names: List[str]) -> List[Tuple[VideoMeta, Array4D]]:
        _names = set(names)
        return [(v.meta, v.content) for v in self.videos if v.meta.name in _names]

    def probability(
        self, phrase: str, videos: List[str]
    ) -> List[Tuple[VideoMeta, List[float]]]:
        for name in videos:
            if name not in self.names:
                raise Exception(f"Video with name '{name}' does not exist")

        files: List[VideoMeta] = []
        features: List[Array2D] = []
        for video in self.videos:
            v = video.meta
            if v.name not in videos:
                continue

            f = video.features
            files.append(v)
            features.append(f)

        np_features = np.concatenate(features)  # type: ignore

        torch_features = torch.tensor(np_features)
        torch_features = normalize_features(torch_features)

        text_tokens = clip.tokenize([phrase]).to(device)  # type: ignore

        with torch.no_grad():
            text_features: torch.Tensor = model.encode_text(text_tokens).float()  # type: ignore
        text_features = normalize_features(text_features)

        similarity: Array2D = (
            torch.matmul(text_features, torch_features.transpose(0, 1)).cpu().numpy()
        )

        index = 0
        ret: List[Tuple[VideoMeta, List[float]]] = []
        for file in files:
            ret.append((file, similarity[0, index : index + file.frames].tolist()))
            index += file.frames
        return ret

    def find(
        self, phrase: str, videos: List[str], num_slice: int, threshold: float
    ) -> List[Tuple[str, Tuple[int, int]]]:
        raise Exception("not implemented")
