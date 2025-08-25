# adapters.py
import base64
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import requests
from coco_labels import COCO_LABEL_TO_ID
from PIL import Image


# ---- moved helpers exactly as-is in logic ----
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


def _hf_json_to_yolo_lines(text: str, image_path: str) -> str:
    """
    Convert Hugging Face inference JSON (list of detections with pixel boxes)
    into YOLO-format lines:
      <class_id> <xc_norm> <yc_norm> <w_norm> <h_norm> <conf>
    """
    j = json.loads(text)
    if not isinstance(j, list):
        return ""

    if not image_path or not os.path.exists(image_path):
        raise RuntimeError("image_path required for Hugging Face output normalization")

    with Image.open(image_path) as im:
        W, H = im.size

    lines = []
    for det in j:
        score = float(det.get("score", 0.0))
        label = str(det.get("label", "")).strip()
        box = det.get("box", {}) or {}
        xmin = float(box.get("xmin", 0.0))
        ymin = float(box.get("ymin", 0.0))
        xmax = float(box.get("xmax", 0.0))
        ymax = float(box.get("ymax", 0.0))

        # clamp
        xmin = max(0.0, min(xmin, W))
        xmax = max(0.0, min(xmax, W))
        ymin = max(0.0, min(ymin, H))
        ymax = max(0.0, min(ymax, H))

        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        cx = xmin + bw / 2.0
        cy = ymin + bh / 2.0

        xc = cx / W if W > 0 else 0.0
        yc = cy / H if H > 0 else 0.0
        w = bw / W if W > 0 else 0.0
        h = bh / H if H > 0 else 0.0

        # map label to COCO id
        cls_id = COCO_LABEL_TO_ID.get(label, -1)
        if cls_id < 0:
            warnings.warn(f"Label '{label}' not in COCO map; mapping to 0")
            cls_id = 0

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {score:.6f}")

    return "\n".join(lines)


def _ultra_json_to_yolo_lines(text: str) -> str:
    """
    Convert Ultralytics predict.ultralytics.com JSON into YOLO lines:
      <class_id> <xc_norm> <yc_norm> <w_norm> <h_norm> <conf>

    Expects payload like:
    {
      "images": [
        {
          "results": [
            {"box": {"x1":..., "x2":..., "y1":..., "y2":...},
             "class": 16, "confidence": 0.68, "name": "dog"}
          ],
          "shape": [H, W],
          ...
        }
      ],
      "metadata": {...}
    }
    """
    try:
        j = json.loads(text)
    except Exception:
        return ""

    images = j.get("images", [])
    if not images or not isinstance(images, list):
        return ""

    # Ultralytics returns one or more images; we concatenate detections from all
    lines: List[str] = []

    for img in images:
        shape = img.get("shape") or []
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            # If shape missing, skip this image
            continue
        H, W = float(shape[0] or 0), float(shape[1] or 0)
        if H <= 0 or W <= 0:
            continue

        results = img.get("results", []) or []
        for det in results:
            box = det.get("box", {}) or {}
            x1 = float(box.get("x1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y1 = float(box.get("y1", 0.0))
            y2 = float(box.get("y2", 0.0))

            # clamp to image bounds
            x1 = max(0.0, min(x1, W))
            x2 = max(0.0, min(x2, W))
            y1 = max(0.0, min(y1, H))
            y2 = max(0.0, min(y2, H))

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0

            xc = cx / W if W > 0 else 0.0
            yc = cy / H if H > 0 else 0.0
            w = bw / W if W > 0 else 0.0
            h = bh / H if H > 0 else 0.0

            cls_id = int(det.get("class", 0))
            conf = float(det.get("confidence", 0.0))

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}")

    return "\n".join(lines)


# ---- new adapter functions (logic unchanged) ----
def prepare_request_body(
    headers: Dict[str, Any],
    input_field: str,
    image_path: str,
    req_kwargs: Dict[str, Any],
    endpoint: str,
    mode: str,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Mutates req_kwargs to include the correct request body based on Content-Type,
    then directly performs the POST + parse by calling post_and_parse.
    """

    # local file → base64
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        data = f.read()
        if not data:
            raise ValueError(f"Image file is empty: {image_path}")
        payload_value = base64.b64encode(data).decode("utf-8")

    ct = headers.get("Content-Type", "")

    if ct == "application/x-www-form-urlencoded":
        # form field: {"image": "<base64string>"}
        req_kwargs[input_field] = payload_value

    elif ct == "multipart/form-data":
        # typical file upload
        filename = os.path.basename(image_path)
        req_kwargs["files"] = {input_field: (filename, open(image_path, "rb"))}

    elif ct.startswith("image/"):
        # raw binary upload (jpg, png, webp, tiff, etc.)
        with open(image_path, "rb") as f:
            req_kwargs[input_field] = f.read()

    elif ct == "application/json":
        req_kwargs["json"] = {input_field: payload_value}

    else:
        if mode != "ultralytics":
            raise ValueError(f"Unsupported Content-Type: {ct}")

    if mode == "ultralytics":
        with open(image_path, "rb") as image_file:
            req_kwargs["files"] = {input_field: image_file}

            resp = requests.post(
                endpoint,
                headers=req_kwargs.get("headers"),
                files=req_kwargs.get("files"),
                data=req_kwargs.get("params"),
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")

    else:
        resp = requests.post(endpoint, **req_kwargs)
        if resp.status_code != 200:
            raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")

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

    elif mode.lower() in ("huggingface", "hf"):
        try:
            j = resp.json()
        except Exception:
            raise RuntimeError(f"HF response not JSON: {resp.text[:200]}")

        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(f"Hugging Face error: {j.get('error')}")

        yolo_text = _hf_json_to_yolo_lines(resp.text, image_path)

    elif mode.lower() in ("ultralytics", "ul"):
        try:
            j = resp.json()
        except Exception:
            raise RuntimeError(f"Ultralytics response not JSON: {resp.text[:200]}")

        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(f"Ultralytics error: {j.get('error')}")

        yolo_text = _ultra_json_to_yolo_lines(resp.text)

    return parse_yolo_predictions(yolo_text if yolo_text is not None else resp.text)
