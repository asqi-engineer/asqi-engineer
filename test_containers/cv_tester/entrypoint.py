#!/usr/bin/env python3
import argparse
import json
import os
import sys

from detectors import build_detector
from dotenv import load_dotenv
from evaluator import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Computer Vision SUT entrypoint")
    parser.add_argument(
        "--systems-params", required=True, help="JSON string with systems params"
    )
    parser.add_argument(
        "--test-params", required=True, help="JSON string with test params"
    )
    args = parser.parse_args()

    try:
        # Parse params
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        # Extract system_under_test
        sut_params = systems_params.get("system_under_test", {})
        if not sut_params:
            raise ValueError("Missing system_under_test in systems_params")

        # Load .env (explicit path wins; fallback to default .env if present)
        env_file = sut_params.get("env_file") or os.getenv("ENV_FILE")
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"[DEBUG] Loaded environment from: {env_file}")
        else:
            load_dotenv()
            print("[DEBUG] Loaded environment from default .env (if present)")

        # Required test params
        img_dir = test_params.get("input_image_path")
        gt_dir = test_params.get("groundtruth_path")
        if not img_dir or not os.path.isdir(img_dir):
            raise ValueError("Invalid or missing input_image_path")
        if not gt_dir or not os.path.isdir(gt_dir):
            raise ValueError("Invalid or missing groundtruth_path")

        # Optional knobs (defaults preserve your original behavior)
        conf_thr = float(sut_params.get("conf_thr", 0.0))
        iou_thr = float(sut_params.get("iou_thr", 0.5))
        use_classes = bool(sut_params.get("use_classes", False))

        # Build detector via adapter factory (Ultralytics/TorchVision/ONNX/etc.)
        print("[DEBUG] Building detector from sut_params...")
        detector = build_detector(sut_params)
        print("[DEBUG] Detector ready.")

        print(
            f"[DEBUG] Starting evaluation | IoU={iou_thr}, conf>={conf_thr}, use_classes={use_classes}"
        )
        map_score, precision, recall, num_images = evaluate_dataset(
            detector=detector,
            img_dir=img_dir,
            gt_dir=gt_dir,
            iou_thr=iou_thr,
            conf_thr=conf_thr,
            use_classes=use_classes,
        )

        result = {
            "success": True,
            "map": round(
                float(map_score), 4
            ),  # NOTE: this is mean IoU of matches (your original "mAP")
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "num_images": int(num_images),
        }
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
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
