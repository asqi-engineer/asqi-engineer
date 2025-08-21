#!/usr/bin/env python3
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


# ---------- I/O helpers ----------
def list_images(img_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(paths)


def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO-format labels (class xc yc w h), normalized to [0,1]."""
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


# ---------- API helpers ----------
def parse_yolo_predictions(
    text: str,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Accepts lines of:
      <class> <xc> <yc> <w> <h> [<conf>]
    Returns list of (cls, xc, yc, w, h, conf). Missing conf -> 1.0.
    """
    preds = []
    for raw in text.strip().splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            if len(parts) == 6:
                c, xc, yc, w, h, conf = parts
                preds.append(
                    (
                        int(float(c)),
                        float(xc),
                        float(yc),
                        float(w),
                        float(h),
                        float(conf),
                    )
                )
            elif len(parts) == 5:
                c, xc, yc, w, h = parts
                preds.append(
                    (int(float(c)), float(xc), float(yc), float(w), float(h), 1.0)
                )
        except Exception:
            continue
    return preds


def inference(
    endpoint: str,
    image_path: str,
    api_params: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Send one image to an external detection API and parse YOLO-style predictions.

    Args:
        endpoint:
            Full URL of the API endpoint, e.g. "http://host.docker.internal:8000/predict".

        image_path:
            Local path to the image file to upload.

        api_params:
            A dict of keyword arguments forwarded **verbatim** to `requests.post(**kwargs)`,
            except for the special key `"input_field"` which controls the multipart field name.
            This gives you full flexibility to match any API without changing code.

            Common keys you can use in `api_params`:
              - input_field (str):
                  The multipart/form-data field name the API expects for the image upload.
                  Examples:
                    * FastAPI: `def predict(file: UploadFile = File(...))`  -> input_field="file"
                    * FastAPI: `def predict(image: UploadFile = File(...))` -> input_field="image"
                  If omitted, defaults to "file".

              - params (dict):
                  Query string parameters appended to the URL. Use this for arguments your API
                  declares as `Query(...)`. Example:
                    api_params = {
                      "params": {"conf": 0.25, "iou": 0.45}
                    }
                  This yields: POST /predict?conf=0.25&iou=0.45

              - data (dict):
                  Extra **form fields** sent alongside the file in the multipart body
                  (server-side read as regular form fields). Example:
                    api_params = {
                      "data": {"model": "yolov8n", "resize": "640"}
                    }

              - headers (dict):
                  HTTP headers (e.g., auth keys/tokens). Example:
                    api_params = {
                      "headers": {"Authorization": "Bearer <token>"}
                    }

    Expected API response body:
        The API **must** return plaintext YOLO-format lines, one detection per line.
        Coordinates must be **normalized** to [0,1].

        Accepted formats (5 or 6 values per line):
          1) "<class> <xc> <yc> <w> <h> <conf>"
          2) "<class> <xc> <yc> <w> <h>"

        Where:
          - class: integer class id (e.g., 0, 1, 27)
          - xc, yc: box center (normalized)
          - w, h:  box width/height (normalized)
          - conf:  optional confidence [0..1] (if absent, defaults to 1.0)

        Examples:
          "0 0.651806 0.579156 0.310504 0.817682 0.903626"
          "27 0.383923 0.554457 0.057366 0.521665 0.885660"
          "0 0.393904 0.500242 0.268452 0.973209"            # no conf -> treated as 1.0

        ⚠️ If your model/server returns another schema (JSON, pixels, absolute coords, etc.),
           put a small proxy in front of it to convert into the exact YOLO text format above.

    Returns:
        List of predictions as tuples:
          [(class_id, xc, yc, w, h, conf), ...]
        with all values as floats except class_id (int).

    Raises:
        RuntimeError: if the HTTP status is not 200.
    """
    params_in = api_params or {}
    # Work on a copy so we don't mutate the caller's dict
    req_kwargs: Dict[str, Any] = dict(params_in)

    # Extract and remove the only special key we care about
    input_field: str = req_kwargs.pop("input_field", "file")

    # Ensure no one overrides 'files' via api_params
    if "files" in req_kwargs:
        req_kwargs.pop("files")

    with open(image_path, "rb") as f:
        files = {
            input_field: (os.path.basename(image_path), f, "application/octet-stream")
        }
        resp = requests.post(endpoint, files=files, **req_kwargs)

    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")

    return parse_yolo_predictions(resp.text)


# ---------- Evaluation ----------
def evaluate_dataset(
    img_dir: str,
    gt_dir: str,
    endpoint: str,
    iou_thr: float = 0.5,
    api_params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float, int]:
    """
    Run inference via external API (endpoint) and evaluate vs YOLO ground-truth labels.
    Returns:
      map_score (mean IoU of matched predictions), precision, recall, num_images
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
            continue
        h, w = img.shape[:2]

        # Ground truth
        label_path = os.path.join(
            gt_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        gt_norm = load_yolo_labels(label_path)
        gt_abs = yolo_to_xyxy_abs(gt_norm, w, h)

        # Predictions from API
        preds = inference(endpoint, img_path, api_params=api_params)

        # Convert preds to abs xyxy
        pred_items: List[Tuple[int, float, float, float, float, float]] = []
        for cls_pred, xc, yc, pw, ph, conf in preds:
            x1 = (xc - pw / 2.0) * w
            y1 = (yc - ph / 2.0) * h
            x2 = (xc + pw / 2.0) * w
            y2 = (yc + ph / 2.0) * h
            pred_items.append((int(cls_pred), x1, y1, x2, y2, conf))

        # Greedy 1-1 matching (class-agnostic)
        matched_gt = set()
        for _, px1, py1, px2, py2, _ in pred_items:
            best_iou = 0.0
            best_idx = -1
            for i, (_, gx1, gy1, gx2, gy2) in enumerate(gt_abs):
                if i in matched_gt:
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
