# detectors.py
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Detector(ABC):
    """
    predict(img) -> (xyxy[N,4], conf[N], cls[N])
    - img: BGR uint8 HxWx3 (OpenCV)
    - xyxy: float32 [x1,y1,x2,y2]
    - conf: float32 [0..1] (use 1.0 if model has no score)
    - cls : int32   class ids matching your GT ids (remap if needed)
    """

    @abstractmethod
    def predict(self, img) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


# -------- Ultralytics YOLO adapter --------
from ultralytics import YOLO


class YoloUltralyticsAdapter(Detector):
    def __init__(self, weights: str, device: str | None = None):
        self.model = YOLO(weights)
        if device:
            self.model.to(device)

    def predict(self, img):
        res = self.model(img, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32)
        cls = res.boxes.cls.cpu().numpy().astype(np.int32)
        return xyxy, conf, cls


# -------- TorchVision Faster R-CNN adapter (example) --------
import torch
import torchvision


class TorchvisionFRCNNAdapter(Detector):
    def __init__(
        self,
        weights: str = "fasterrcnn_resnet50_fpn",
        device: str | None = None,
        score_thresh: float = 0.0,
    ):
        if weights == "fasterrcnn_resnet50_fpn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT"
            )
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=None, num_classes=91
            )
            state = torch.load(weights, map_location="cpu")
            self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.score_thresh = float(score_thresh)

    def predict(self, img):
        x = torch.from_numpy(img[:, :, ::-1]).to(self.device)  # BGR->RGB
        x = x.permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self.model([x])[0]
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        keep = scores >= self.score_thresh
        boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)[keep]
        scores = scores[keep]
        labels = out["labels"].detach().cpu().numpy().astype(np.int32)[keep]
        return boxes, scores, labels


def build_detector(sut_params: dict) -> Detector:
    """
    sut_params example:
      {"type":"ultralytics","model":"yolov8s.pt"}
      {"type":"torchvision","model":"fasterrcnn_resnet50_fpn"}
    If 'type' missing, defaults to Ultralytics.
    """
    det_type = str(sut_params.get("type", "ultralytics")).lower()
    device = sut_params.get("device")

    if det_type in ("ultralytics", "yolo", "yolov5", "yolov8"):
        weights = sut_params.get("model", "yolov8n.pt")
        return YoloUltralyticsAdapter(weights=weights, device=device)

    if det_type in ("torchvision", "frcnn", "fasterrcnn"):
        weights = sut_params.get("model", "fasterrcnn_resnet50_fpn")
        score = float(sut_params.get("conf_thr", 0.0))
        return TorchvisionFRCNNAdapter(
            weights=weights, device=device, score_thresh=score
        )

    raise ValueError(f"Unknown detector type: {det_type}")
