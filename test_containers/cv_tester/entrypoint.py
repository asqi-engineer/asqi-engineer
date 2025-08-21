#!/usr/bin/env python3
import argparse
import json
import os
import sys

from evaluator import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="CV SUT entrypoint (API-based, with full endpoint)"
    )
    parser.add_argument(
        "--sut-params", required=True, help="JSON string with SUT params"
    )
    parser.add_argument(
        "--test-params", required=True, help="JSON string with test params"
    )
    args = parser.parse_args()

    try:
        # Parse params
        sut_params = json.loads(args.sut_params)
        test_params = json.loads(args.test_params)
        if not sut_params or not test_params:
            raise ValueError("Missing SUT or test parameters")

        # Validate SUT type
        sut_type = sut_params.get("type")
        if sut_type not in ["object_detection_api"]:
            raise ValueError(f"Unsupported SUT type: {sut_type}")

        # Required: full endpoint URL
        endpoint = sut_params.get("endpoint")
        if not endpoint:
            raise ValueError("Missing 'endpoint' in sut_params (must be full URL)")

        # Grouped API params (must include input_field)
        api_params = sut_params.get("api_params", {})
        if "input_field" not in api_params or not api_params["input_field"]:
            raise ValueError(
                "api_params must include 'input_field', which specifies the multipart form field "
                "name used to upload the image. For example, if your API endpoint is defined as "
                "`def predict(file: UploadFile = File(...))`, then use input_field='file'. "
                "If it expects `image: UploadFile`, set input_field='image'."
            )

        # Evaluation knobs
        iou_thr = float(sut_params.get("iou_thr", 0.5))

        # Required test params
        img_dir = test_params.get("input_image_path")
        gt_dir = test_params.get("groundtruth_path")
        if not img_dir or not os.path.isdir(img_dir):
            raise ValueError("Invalid or missing input_image_path")
        if not gt_dir or not os.path.isdir(gt_dir):
            raise ValueError("Invalid or missing groundtruth_path")

        print(f"[DEBUG] Evaluating via API endpoint: {endpoint}")
        print(f"[DEBUG] IoU={iou_thr}")

        map_score, precision, recall, num_images = evaluate_dataset(
            img_dir=img_dir,
            gt_dir=gt_dir,
            endpoint=endpoint,
            iou_thr=iou_thr,
            api_params=api_params,
        )

        result = {
            "success": True,
            "map": round(float(map_score), 4),
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
