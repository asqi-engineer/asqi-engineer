#!/usr/bin/env python3
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


# ---------- I/O helpers ----------
def list_images(img_dir: str) -> List[str]:
    """List images with common extensions."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(paths)


def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO-format labels (class xc yc w h), normalized to [0,1].
    Returns list of tuples: (cls, xc, yc, w, h).
    """
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            boxes.append((int(cls), xc, yc, w, h))
    return boxes


def yolo_to_xyxy_abs(
    boxes: List[Tuple[int, float, float, float, float]],
    img_w: int,
    img_h: int,
) -> List[Tuple[int, float, float, float, float]]:
    """Convert YOLO normalized (xc,yc,w,h) to absolute xyxy with class."""
    converted = []
    for cls, xc, yc, w, h in boxes:
        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h
        converted.append((cls, x1, y1, x2, y2))
    return converted


# ---------- Geometry ----------


def iou_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------- Evaluation ----------


def evaluate_dataset(
    detector,
    img_dir: str,
    gt_dir: str,
    iou_thr: float = 0.5,
    conf_thr: float = 0.0,
    use_classes: bool = False,
) -> Tuple[float, float, float, int]:
    """
    Run inference with the provided detector and evaluate vs YOLO ground-truth labels.

    Returns:
      - map_score: mean IoU of matched predictions (kept to match your original behavior)
      - precision: TP / (TP + FP)
      - recall: TP / (TP + FN)
      - num_images: number of images evaluated
    """
    img_paths = list_images(img_dir)
    if not img_paths:
        raise ValueError(f"No images found in {img_dir}")

    TP = 0
    FP = 0
    FN = 0
    ious: List[float] = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            # Skip unreadable images
            continue
        h, w = img.shape[:2]

        # Ground truth: convert normalized YOLO txt -> absolute xyxy
        label_path = os.path.join(
            gt_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        gt_norm = load_yolo_labels(label_path)
        gt_abs = yolo_to_xyxy_abs(gt_norm, w, h)  # List[(cls, x1,y1,x2,y2)]

        # Predictions: generic interface (xyxy[N,4], conf[N], cls[N])
        xyxy, conf, cls = detector.predict(img)

        # Build prediction items (apply confidence threshold)
        pred_items: List[Tuple[int, float, float, float, float, float]] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if c >= conf_thr:
                pred_items.append(
                    (int(k), float(x1), float(y1), float(x2), float(y2), float(c))
                )

        # Greedy one-to-one matching at IoU threshold (class-aware optional)
        matched_gt = set()
        for p_cls, px1, py1, px2, py2, _ in pred_items:
            best_iou = 0.0
            best_idx = -1
            for i, (g_cls, gx1, gy1, gx2, gy2) in enumerate(gt_abs):
                if i in matched_gt:
                    continue
                if use_classes and (p_cls != g_cls):
                    continue
                iou_val = iou_xyxy((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou_val >= iou_thr and iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_idx >= 0:
                TP += 1
                matched_gt.add(best_idx)
                ious.append(best_iou)
            else:
                FP += 1

        FN += len(gt_abs) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    map_score = float(np.mean(ious)) if ious else 0.0

    return map_score, precision, recall, len(img_paths)
