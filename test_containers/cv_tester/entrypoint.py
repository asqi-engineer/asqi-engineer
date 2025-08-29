#!/usr/bin/env python3
"""
===============================================================================
CV SUT Entrypoint (API-based evaluation with YOLO-compatible format)
===============================================================================

This script is the **entrypoint** for evaluating computer vision Systems Under Test (SUTs).
It connects to external inference APIs (Hugging Face, Roboflow, Ultralytics, or local endpoints),
runs predictions on a dataset of images, and compares them against YOLO-format ground-truth labels.

-------------------------------------------------------------------------------
How it works
-------------------------------------------------------------------------------
1. Loads system parameters (SUT) and test parameters (dataset) from JSON strings:
     --sut-params '{"type":"object_detection_api", ...}'
     --test-params '{"input_image_path":"...", "groundtruth_path":"..."}'

2. Reads API credentials and endpoints from environment variables:
   - Copy `.env.example` → `.env` and fill in API keys and endpoints.
   - Examples:
       ROBOFLOW_API_KEY, ROBOFLOW_ENDPOINT
       HUGGINGFACE_API_KEY, HUGGINGFACE_ENDPOINT
       ULTRALYTICS_API_KEY, ULTRALYTICS_ENDPOINT
       LOCAL_ENDPOINT

3. Selects which API mode to use (`mode` field in sut_params):
   - "roboflow"     → Roboflow API
   - "huggingface"  → Hugging Face Inference API
   - "ultralytics"  → Ultralytics hosted API
   - "local"        → Custom/local endpoint

4. Calls the selected API for each image in the dataset.
   - The API must accept image uploads via the `input_field` specified in `api_params`
     (e.g., `file` or `image` in a FastAPI endpoint).
   - The API response is parsed and converted into YOLO-format predictions.

5. Evaluates predictions vs ground truth:
   - Computes precision, recall, mean IoU, and number of images.

-------------------------------------------------------------------------------
Where configs are defined
-------------------------------------------------------------------------------
- `suts/` folder:
    Contains definitions of supported SUTs and parameters.
- `cv_tester_suts.yaml`:
    Defines available SUTs you can run tests against.
- `suites/cv_tester_suite.yaml`:
    Defines test suites (datasets + configs) that call into the SUTs.

-------------------------------------------------------------------------------
Prediction result formats
-------------------------------------------------------------------------------
Our **native format** is YOLO text lines:
    <class> <xc> <yc> <w> <h> [<conf>]

However, we also try to adapt common API response formats automatically:
  ✓ Raw YOLO text
  ✓ COCO-like JSON (list of {bbox, category_id, score})
  ✓ LabelMe JSON (imageWidth, imageHeight, shapes with points)
  ✓ Pascal VOC XML (<size>, <object><bndbox>)

Special cases (handled internally):
  - Roboflow API JSON
  - Hugging Face inference API JSON
  - Ultralytics predict API JSON

If your API outputs another format, it may not be supported yet.
Check `adapters.py` for the parsing logic, and extend it if needed.

-------------------------------------------------------------------------------
Summary
-------------------------------------------------------------------------------
- Supports evaluation of SUTs via Hugging Face, Roboflow, Ultralytics, or custom/local APIs.
- Ground truth labels must be in YOLO format.
- Predictions are auto-parsed into YOLO format wherever possible.
- Outputs metrics: mAP (mean IoU of matched boxes), precision, recall, and dataset size.

===============================================================================
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
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
        load_dotenv()

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
        mode = sut_params.get("mode", "local")
        api_params = sut_params.get("api_params", {})

        # Handle Roboflow mode: inject API key + endpoint from .env
        if mode == "roboflow":
            rf_key = os.getenv("ROBOFLOW_API_KEY")
            rf_endpoint = os.getenv("ROBOFLOW_ENDPOINT")

            if not rf_key:
                raise ValueError("ROBOFLOW_API_KEY not found in environment")

            if not rf_endpoint:
                raise ValueError("ROBOFLOW_ENDPOINT not found in environment")

            # Inject/overwrite api_key
            if "params" not in api_params:
                api_params["params"] = {}
            api_params["params"]["api_key"] = rf_key

            # Overwrite endpoint
            endpoint = rf_endpoint

        elif mode == "huggingface":
            hf_token = os.getenv("HUGGINGFACE_API_KEY")
            hf_endpoint = os.getenv("HUGGINGFACE_ENDPOINT")

            if not hf_token:
                raise ValueError(
                    "HF_TOKEN (or HUGGINGFACE_API_KEY) not found in environment"
                )

            if not hf_endpoint:
                raise ValueError("HUGGINGFACE_ENDPOINT not found in environment")

            endpoint = hf_endpoint
            if not isinstance(api_params, dict):
                api_params = {}

            headers = dict(api_params.get("headers", {}) or {})
            auth_val = headers.get("Authorization")
            if auth_val:
                if "${HUGGINGFACE_API_KEY}" in auth_val:
                    if not hf_token:
                        raise ValueError("HUGGINGFACE_API_KEY not set in environment")
                    headers["Authorization"] = auth_val.replace(
                        "${HUGGINGFACE_API_KEY}", hf_token
                    )
            else:
                if not hf_token:
                    raise ValueError(
                        "HF_TOKEN or HUGGINGFACE_API_KEY not found in environment"
                    )
                headers["Authorization"] = f"Bearer {hf_token}"

            api_params["headers"] = headers

        elif mode == "ultralytics":
            ul_token = os.getenv("ULTRALYTICS_API_KEY")
            ul_endpoint = os.getenv("ULTRALYTICS_ENDPOINT")

            if not ul_token:
                raise ValueError("ULTRALYTICS_API_KEY not found in environment")
            if not ul_endpoint:
                raise ValueError("ULTRALYTICS_ENDPOINT not found in environment")

            endpoint = ul_endpoint
            if not isinstance(api_params, dict):
                api_params = {}
            headers = dict(api_params.get("headers", {}) or {})

            x_api_key_val = headers.get("x-api-key")

            if x_api_key_val:
                if "${ULTRALYTICS_API_KEY}" in x_api_key_val:
                    headers["x-api-key"] = x_api_key_val.replace(
                        "${ULTRALYTICS_API_KEY}", ul_token
                    )
            else:
                raise ValueError(
                    "api_params.headers must include 'x-api-key' for Ultralytics API"
                )
            api_params["headers"] = headers

        else:  # mode == "local"
            endpoint = os.getenv("LOCAL_ENDPOINT")

        # Final endpoint validation (after possible override)
        if not endpoint:
            raise ValueError(
                "Missing 'endpoint' (sut_params or ROBOFLOW_ENDPOINT must provide a full URL)"
            )

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
            mode=mode,
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
