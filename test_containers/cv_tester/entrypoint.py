import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def load_yolo_labels(label_path):
    """
    Load YOLO-format labels from a txt file.
    Format per line: class x_center y_center width height (all normalized)
    Returns list of [class, x1, y1, x2, y2] in absolute pixel coords.
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
            boxes.append([int(cls), xc, yc, w, h])
    return boxes


def yolo_to_xyxy(boxes, img_w, img_h):
    """Convert YOLO normalized format to absolute [x1, y1, x2, y2]."""
    converted = []
    for cls, xc, yc, w, h in boxes:
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        converted.append([cls, x1, y1, x2, y2])
    return converted


def iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate_dataset(model, img_dir, gt_dir, iou_thr=0.5):
    """
    Run inference with YOLO model and evaluate vs YOLO ground-truth labels.
    Returns mAP@0.5, precision, recall, num_images.
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        raise ValueError(f"No images found in {img_dir}")

    TP, FP, FN = 0, 0, 0
    all_ious = []

    for img_path in img_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # ground truth
        label_path = os.path.join(
            gt_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        gt_boxes = load_yolo_labels(label_path)
        gt_boxes = yolo_to_xyxy(gt_boxes, w, h)

        # predictions
        results = model(img, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            pred_boxes.append([x1, y1, x2, y2])

        matched = set()
        for pb in pred_boxes:
            found_match = False
            for i, gb in enumerate(gt_boxes):
                iou_val = iou(pb, gb[1:])
                if iou_val >= iou_thr and i not in matched:
                    TP += 1
                    matched.add(i)
                    all_ious.append(iou_val)
                    found_match = True
                    break
            if not found_match:
                FP += 1
        FN += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    map_score = np.mean(all_ious) if all_ious else 0.0

    return map_score, precision, recall, len(img_paths)


def main():
    """
    Object detection evaluator.
    - Consumes:
        --sut-params  : JSON (validated but unused here)
        --test-params : JSON with:
            input_image_path (str, required)
            groundtruth_path (str, required)
    - Produces:
        JSON with success, map, precision, recall, num_images.
    """
    parser = argparse.ArgumentParser(description="YOLO Object Detection Evaluator")
    parser.add_argument(
        "--sut-params", required=True, help="SUT parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )
    args = parser.parse_args()

    try:
        print("[DEBUG] Parsing parameters...")
        sut_params = json.loads(args.sut_params)
        test_params = json.loads(args.test_params)
        print(f"[DEBUG] SUT params: {sut_params}")
        print(f"[DEBUG] Test params: {test_params}")

        # Required test params
        img_dir = test_params.get("input_image_path")
        gt_dir = test_params.get("groundtruth_path")
        if not img_dir or not os.path.isdir(img_dir):
            raise ValueError("Invalid or missing input_image_path")
        if not gt_dir or not os.path.isdir(gt_dir):
            raise ValueError("Invalid or missing groundtruth_path")

        print(f"[DEBUG] Input images directory: {img_dir}")
        print(f"[DEBUG] Ground truth labels directory: {gt_dir}")

        # Load YOLO model (replace with your weight if needed)
        model_name = sut_params.get("model", "yolov8n.pt")
        print(f"[DEBUG] Loading YOLO model: {model_name} ...")
        model = YOLO(model_name)
        print("[DEBUG] Model loaded successfully.")

        # Run evaluation
        print("[DEBUG] Starting evaluation...")
        map_score, precision, recall, num_images = evaluate_dataset(
            model, img_dir, gt_dir
        )

        print("[DEBUG] Evaluation completed.")
        print(
            f"[DEBUG] mAP: {map_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Num images: {num_images}"
        )

        result = {
            "success": True,
            "map": float(round(map_score, 4)),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "num_images": int(num_images),
        }

        print("[DEBUG] Final result JSON:")
        print(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "map": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_images": 0,
        }
        print("[ERROR]", str(e))
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
