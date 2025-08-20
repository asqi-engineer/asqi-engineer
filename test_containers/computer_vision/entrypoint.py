import argparse
import json
import os
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(gt_path, pred_path):
    """
    Evaluate COCO-format ground truth and predictions.
    Returns mAP, precision, recall, num_images.
    """
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    map_score = float(coco_eval.stats[0])  # mAP IoU=0.50:0.95

    # Precision and recall approximations
    precision = float(coco_eval.stats[0])  # use mAP as proxy for precision
    recall = float(coco_eval.stats[8])  # AR@100 as recall

    num_images = len(coco_gt.getImgIds())

    return map_score, precision, recall, num_images


def main():
    """
    Object detection evaluator.
    - Consumes:
        --sut-params  : JSON (validated but unused here)
        --test-params : JSON with:
            groundtruth_path (str, required)
            prediction_path  (str, required)
    - Produces:
        JSON with success, map, precision, recall, num_images, echo of paths.
    """
    parser = argparse.ArgumentParser(description="COCO Object Detection Evaluator")
    parser.add_argument(
        "--sut-params",
        required=True,
        help="SUT parameters as JSON string (kept for API parity)",
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )
    args = parser.parse_args()

    try:
        sut_params = json.loads(args.sut_params)  # not used but validated as JSON
        test_params = json.loads(args.test_params)

        # Validate SUT type
        sut_type = sut_params.get("type")
        if sut_type not in ["computer_vision"]:
            raise ValueError(f"Unsupported SUT type: {sut_type}")

        # Required test params
        gt_path = test_params.get("groundtruth_path")
        pred_path = test_params.get("prediction_path")
        if not gt_path or not isinstance(gt_path, str):
            raise ValueError("Missing or invalid test param: groundtruth_path")
        if not pred_path or not isinstance(pred_path, str):
            raise ValueError("Missing or invalid test param: prediction_path")

        # Ensure files exist
        if not os.path.exists(gt_path):
            raise ValueError(f"Ground truth file not found: {gt_path}")
        if not os.path.exists(pred_path):
            raise ValueError(f"Prediction file not found: {pred_path}")
        if os.path.isdir(gt_path) or os.path.isdir(pred_path):
            raise ValueError(
                "Expected JSON files, but got directories. Pass full file paths."
            )

        # Real COCO evaluation
        map_score, precision, recall, num_images = evaluate_coco(gt_path, pred_path)

        result = {
            "success": True,
            "map": round(map_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "num_images": num_images,
            "groundtruth_path": gt_path,
            "prediction_path": pred_path,
        }

        print(json.dumps(result, indent=2))
        sys.exit(0)

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "map": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_images": 0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "map": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_images": 0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
