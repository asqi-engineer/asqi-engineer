# adapters.py
import base64
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import requests
from coco_labels import COCO_LABEL_TO_ID
from defusedxml import ElementTree as ET
from PIL import Image


def _looks_like_yolo_text(s: str) -> bool:
    for line in s.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = t.split()
        if len(parts) in (5, 6):
            try:
                int(float(parts[0]))
                for x in parts[1:]:
                    float(x)
                return True
            except (ValueError, TypeError) as e:
                raise RuntimeError(f"Invalid YOLO line format: {parts}") from e
    return False


def _extract_dims_anywhere(j: dict) -> tuple[float, float] | None:
    """
    Find image dimensions inside a JSON payload.
    Input: dict (possibly nested) with fields like imageWidth/imageHeight or width/height.
    Output: (W, H) as floats if found, else None.
    """

    # LabelMe style
    if j.get("imageWidth") and j.get("imageHeight"):
        return float(j["imageWidth"]), float(j["imageHeight"])
    # top-level width/height
    if j.get("width") and j.get("height"):
        return float(j["width"]), float(j["height"])
    # nested dicts
    for v in j.values():
        if isinstance(v, dict):
            dims = _extract_dims_anywhere(v)
            if dims:
                return dims
    return None


def _bbox_is_xyxy(b: list[float]) -> bool:
    """
    Check if a bbox is in [x1, y1, x2, y2] format.
    Input: list of 4 floats [x1, y1, x2, y2].
    Output: True if x2 >= x1 and y2 >= y1, else False.
    """

    if len(b) != 4:
        return False
    x1, y1, x2, y2 = b
    return x2 >= x1 and y2 >= y1


def _norm_from_xyxy(
    b: list[float], W: float, H: float
) -> tuple[float, float, float, float]:
    """
    Normalize an [x1, y1, x2, y2] bbox to YOLO format.
    Input: bbox [x1, y1, x2, y2], image width W, height H.
    Output: (xc, yc, w, h) normalized to [0,1].
    """

    x1, y1, x2, y2 = b
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / W, cy / H, bw / W, bh / H


def _parse_coco_like_list(
    arr: list, top_dims: tuple[float, float] | None
) -> str | None:
    """
    Parse COCO-like detection list into YOLO lines.
    Input: list of dicts with "bbox":[x,y,w,h] or [x1,y1,x2,y2], plus "category_id"/"class"/"label", optional "score".
           top_dims = (W,H) if not in each dict.
    Output: YOLO-format string or None if no detections.
    """
    lines = []
    for det in arr:
        if not isinstance(det, dict) or "bbox" not in det:
            continue
        bbox = det["bbox"]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        # dimensions
        W = det.get("image_width") or (top_dims[0] if top_dims else None)
        H = det.get("image_height") or (top_dims[1] if top_dims else None)
        if not W or not H or W <= 0 or H <= 0:
            continue
        b = list(map(float, bbox))
        if _bbox_is_xyxy(b):
            xc, yc, w, h = _norm_from_xyxy(b, W, H)
        else:
            x, y, w_, h_ = b
            xc, yc, w, h = _norm_from_xyxy([x, y, x + w_, y + h_], W, H)
        cls_id = det.get("category_id", det.get("class", 0))
        try:
            cls_id = int(cls_id)
        except Exception:
            cls_id = 0
        score = float(det.get("score", 1.0))
        lines.append(
            f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {max(0, min(1, score)):.6f}"
        )
    return "\n".join(lines) if lines else None


def _parse_labelme_like(j: dict) -> str | None:
    """
    Parse LabelMe JSON into YOLO lines.
    Input: dict with "imageWidth","imageHeight","shapes":[{"points":[[x,y],...],"label":...},...].
    Output: YOLO-format string or None if invalid.
    """
    if not (
        j.get("imageWidth")
        and j.get("imageHeight")
        and isinstance(j.get("shapes"), list)
    ):
        return None
    W, H = float(j["imageWidth"]), float(j["imageHeight"])
    lines = []
    for s in j["shapes"]:
        pts = s.get("points")
        if not (isinstance(pts, list) and len(pts) >= 2):
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        xc, yc, w, h = _norm_from_xyxy([x1, y1, x2, y2], W, H)
        label = s.get("label", 0)
        try:
            cid = int(label)
        except Exception:
            cid = 0
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} 1.000000")
    return "\n".join(lines) if lines else None


def _parse_voc_xml_text(xml_text: str) -> str | None:
    """
    Parse Pascal VOC XML into YOLO lines.
    Input: XML string with <size><width>,<height> and <object><bndbox>.
    Output: YOLO-format string or None if invalid.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    W = H = None
    sz = root.find(".//size")
    if sz is not None:
        try:
            W = float(sz.findtext("width") or 0)
            H = float(sz.findtext("height") or 0)
        except (ValueError, TypeError, AttributeError) as e:
            raise RuntimeError("Invalid width/height values in VOC XML") from e
    if not W or not H or W <= 0 or H <= 0:
        return None
    lines = []
    for obj in root.findall(".//object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        try:
            x1 = float(b.findtext("xmin") or 0)
            y1 = float(b.findtext("ymin") or 0)
            x2 = float(b.findtext("xmax") or 0)
            y2 = float(b.findtext("ymax") or 0)
        except (ValueError, TypeError, AttributeError) as e:
            raise RuntimeError("Invalid bounding box values in VOC XML") from e
        xc, yc, w, h = _norm_from_xyxy([x1, y1, x2, y2], W, H)
        name = obj.findtext("name") or "0"
        try:
            cid = int(name)
        except Exception:
            cid = 0
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} 1.000000")
    return "\n".join(lines) if lines else None


def auto_to_yolo_lines_from_text(text: str) -> str:
    """
    Auto-parse response text into YOLO lines.
    Supported:
      - Raw YOLO text
      - COCO-like JSON (list of dicts with 'bbox')
      - LabelMe JSON (imageWidth,imageHeight,shapes)
      - Pascal VOC XML
    Else: raise ValueError.
    """
    if not text or not text.strip():
        raise ValueError("Empty response body")

    # Raw YOLO
    if _looks_like_yolo_text(text):
        return text

    # Try JSON
    try:
        j = json.loads(text)
    except Exception:
        j = None

    if isinstance(j, dict):
        # LabelMe
        out = _parse_labelme_like(j)
        if out:
            return out

        # COCO-like possibly nested
        dims = _extract_dims_anywhere(j)
        for k in ("results", "predictions", "detections", "annotations"):
            arr = j.get(k)
            if isinstance(arr, list) and arr:
                out = _parse_coco_like_list(arr, dims)
                if out:
                    return out

        # Single dict with bbox
        if "bbox" in j:
            out = _parse_coco_like_list([j], dims)
            if out:
                return out

    if isinstance(j, list):
        out = _parse_coco_like_list(j, None)
        if out:
            return out

    # Try Pascal VOC XML
    out = _parse_voc_xml_text(text)
    if out:
        return out

    # Otherwise fail
    raise ValueError(
        "Unrecognized response format (not YOLO, COCO-like JSON, LabelMe JSON, or VOC XML)"
    )


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
        raise ValueError(f"Unsupported Content-Type: {ct}")

    timeout_val = float(req_kwargs.pop("timeout", 30.0))

    if mode == "ultralytics":
        resp = requests.post(
            endpoint,
            headers=req_kwargs.get("headers"),
            files=req_kwargs.get("files"),
            data=req_kwargs.get("params"),
            timeout=timeout_val,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")

    else:
        resp = requests.post(endpoint, timeout=timeout_val, **req_kwargs)
        if resp.status_code != 200:
            raise RuntimeError(f"API {resp.status_code}: {resp.text[:200]}")

    yolo_text = ""

    if mode.lower() in ("roboflow", "rf"):
        # Roboflow returns JSON with "predictions" and "image" keys
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
        # Hugging Face Inference API → list of detections with pixel coords
        try:
            j = resp.json()
        except Exception:
            raise RuntimeError(f"HF response not JSON: {resp.text[:200]}")

        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(f"Hugging Face error: {j.get('error')}")

        yolo_text = _hf_json_to_yolo_lines(resp.text, image_path)

    elif mode.lower() in ("ultralytics", "ul"):
        # Ultralytics inference API → JSON with "images" key
        try:
            j = resp.json()
        except Exception:
            raise RuntimeError(f"Ultralytics response not JSON: {resp.text[:200]}")

        if isinstance(j, dict) and "error" in j:
            raise RuntimeError(f"Ultralytics error: {j.get('error')}")

        yolo_text = _ultra_json_to_yolo_lines(resp.text)

    else:
        # Default: try to auto-parse response text
        yolo_text = auto_to_yolo_lines_from_text(resp.text)

    return parse_yolo_predictions(yolo_text)
