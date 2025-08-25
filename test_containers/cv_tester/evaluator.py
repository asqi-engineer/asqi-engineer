#!/usr/bin/env python3

import base64
import glob
import json
import os
import warnings
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
def _parse_yolo_line(
    line: str,
) -> Optional[Tuple[int, float, float, float, float, float]]:
    """
    Parse one YOLO line:
      <class> <xc> <yc> <w> <h> [<conf>]
    Returns (cls, xc, yc, w, h, conf) or None if invalid.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    parts = s.split()
    n = len(parts)
    if n not in (5, 6):
        return None

    try:
        if n == 6:
            c, xc, yc, w, h, conf = parts
            return (
                int(float(c)),
                float(xc),
                float(yc),
                float(w),
                float(h),
                float(conf),
            )
        else:  # n == 5
            c, xc, yc, w, h = parts
            return (int(float(c)), float(xc), float(yc), float(w), float(h), 1.0)
    except (ValueError, TypeError):
        return None


def parse_yolo_predictions(
    text: str,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Accepts lines of:
      <class> <xc> <yc> <w> <h> [<conf>]
    Returns list of (cls, xc, yc, w, h, conf). Missing conf -> 1.0.
    """
    preds: List[Tuple[int, float, float, float, float, float]] = []
    for raw in text.splitlines():
        parsed = _parse_yolo_line(raw)
        if parsed is not None:
            preds.append(parsed)
    return preds


def _rf_json_to_yolo_lines(text: str) -> str:
    """
    Convert Roboflow JSON (center coords in pixels) to YOLO lines:
      <class_id> <xc_norm> <yc_norm> <w_norm> <h_norm> <conf>
    """
    j = json.loads(text)
    iw = float(j.get("image", {}).get("width", 0) or 0)
    ih = float(j.get("image", {}).get("height", 0) or 0)
    if iw <= 0 or ih <= 0:
        return ""

    lines = []
    for p in j.get("predictions", []):
        cx = float(p["x"])
        cy = float(p["y"])
        bw = float(p["width"])
        bh = float(p["height"])
        # normalize to 0..1 (YOLO format expects normalized center/size)
        xc = cx / iw
        yc = cy / ih
        w = bw / iw
        h = bh / ih
        # clamp just in case
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        cls_id = int(p.get("class_id", 0))
        conf = float(p.get("confidence", 0.0))
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}")
    return "\n".join(lines)


def inference(
    endpoint: str,
    mode: str,
    image_path: str,
    api_params: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Send one image to an external detection API and parse predictions into YOLO format.

    Args:
        endpoint (str):
            Full URL of the API endpoint (e.g. "http://localhost:8000/predict").

        mode (str):
            API mode. Special handling is applied for:
              - "roboflow" / "rf":
                  Expects Roboflow JSON response with "predictions" and "image" keys.
                  Converts to YOLO lines internally.
              - any other value:
                  Expects the server to return YOLO-format plaintext directly.

        image_path (str):
            Local path to the image file.

        api_params (dict, optional):
            Dictionary of request options. Keys:
              - "input_field" (str, default="image"):
                    Field name for the uploaded image.
              - "params" (dict):
                    Query string arguments appended to the endpoint.
              - "headers" (dict):
                    Extra HTTP headers (e.g. Content-Type, Authorization).
              - "timeout" (float):
                    Request timeout in seconds (default 30.0).
              - Any other keys are forwarded into `requests.post`.

            Request body depends on headers:
              - If headers["Content-Type"] == "application/x-www-form-urlencoded":
                    Sends image base64 string in form field.
              - If headers["Content-Type"] == "multipart/form-data":
                    Sends image as a file upload.
              - Otherwise (default):
                    Sends JSON body {input_field: "<base64 string>"}.

    Expected API response:
        - For "local" or custom APIs:
              Plaintext YOLO lines, one per detection:
                  "<class> <xc> <yc> <w> <h> [<conf>]"
              Coordinates normalized to [0,1]. Confidence optional (default 1.0).
        - For "roboflow" mode:
              JSON with structure:
                  {"predictions": [...], "image": {"width": W, "height": H}}
              Each prediction must include fields "x", "y", "width", "height",
              "class_id", and "confidence". Converted automatically to YOLO lines.

    Returns:
        List of detections as tuples:
            [(class_id, xc, yc, w, h, conf), ...]
        where coords are normalized floats and class_id is int.

    Raises:
        FileNotFoundError: if image_path does not exist.
        ValueError: if image is empty or response cannot be parsed.
        RuntimeError: if the API returns non-200 status or explicit error JSON.
    """

    params_in = dict(api_params or {})

    # Controls (support timeout at top-level OR inside nested params)
    input_field: str = params_in.pop("input_field", "image")
    nested_params = params_in.pop("params", {}) or {}
    timeout: float = float(params_in.pop("timeout", nested_params.pop("timeout", 30.0)))

    # headers + query params
    headers = params_in.pop("headers", {}) or {}
    query_params: Dict[str, Any] = {**params_in, **nested_params}

    req_kwargs: Dict[str, Any] = {"params": dict(query_params), "timeout": timeout}
    if headers:
        req_kwargs["headers"] = dict(headers)

    # local file → base64
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        data = f.read()
        if not data:
            raise ValueError(f"Image file is empty: {image_path}")
        payload_value = base64.b64encode(data).decode("utf-8")

    # body: raw base64 for Roboflow (form-urlencoded) OR JSON for local
    ct = headers.get("Content-Type", "")
    if ct == "application/x-www-form-urlencoded":
        req_kwargs[input_field] = payload_value
    elif ct == "multipart/form-data":
        filename = os.path.basename(image_path)
        req_kwargs["files"] = {input_field: (filename, open(image_path, "rb"))}
    else:
        # default: JSON body {input_field: "<base64>"}
        req_kwargs["json"] = {input_field: payload_value}

    resp = requests.post(endpoint, **req_kwargs)
    if resp.status_code != 200:
        raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")

    # only convert when mode indicates Roboflow
    yolo_text = None
    if mode.lower() in ("roboflow", "rf"):
        content_type = resp.headers.get("Content-Type", "")
        try:
            if "application/json" not in content_type:
                raise ValueError(f"Non-JSON Content-Type: {content_type}")
            j = resp.json()
            if isinstance(j, dict) and "error" in j:
                raise RuntimeError(f"Remote API error: {j.get('error')}")
            if isinstance(j, dict) and "predictions" in j and "image" in j:
                yolo_text = _rf_json_to_yolo_lines(resp.text)
            else:
                warnings.warn(
                    "JSON received but missing expected keys ('predictions','image'); "
                    "falling back to raw text."
                )
        except ValueError as e:
            snippet = resp.text[:200].replace("\n", " ")
            warnings.warn(
                f"Failed to parse JSON ({e}). Content-Type='{content_type}'. "
                f"Body starts with: {snippet!r}"
            )

    return parse_yolo_predictions(yolo_text if yolo_text is not None else resp.text)


# ---------- Evaluation ----------
def evaluate_dataset(
    img_dir: str,
    gt_dir: str,
    endpoint: str,
    mode: str,
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
        preds = inference(endpoint, mode, img_path, api_params=api_params)

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
