import argparse
import html
import io
import json
import os
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path

import requests as http_requests
import torch
from asqi.datasets import load_hf_dataset
from huggingface_hub import InferenceClient
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def log(level: str, message: str) -> None:
    """Log message to stderr."""
    print(f"[{level}] {message}", file=sys.stderr)


def normalize_label_map(label_map: dict) -> dict:
    """Convert {id: name} to {name: id} for label lookups."""
    if not label_map:
        return {}
    first_val = next(iter(label_map.values()), None)
    if isinstance(first_val, str):
        return {
            v.lower(): (int(k) if isinstance(k, str) else k)
            for k, v in label_map.items()
        }
    return {k.lower(): v for k, v in label_map.items()}


def parse_config(sys_params: dict, test_params: dict) -> dict:
    """Extract configuration from system and test parameters."""
    sut = sys_params.get("system_under_test", {})
    if sut.get("type") != "hf_inference_api":
        raise ValueError(f"Expected hf_inference_api, got: {sut.get('type')}")

    ds_config = test_params.get("input_datasets", {}).get("evaluation_data", {})

    return {
        "model_id": sut.get("model_id", "facebook/detr-resnet-50"),
        "api_key": sut.get("api_key") or os.environ.get("HF_TOKEN"),
        "timeout": sut.get("timeout", 30.0),
        "ds_config": ds_config,
        "label_map": ds_config.get("label_map") or test_params.get("label_map", {}),
        "bbox_format": ds_config.get("bbox_format")
        or test_params.get("bbox_format", "xyxy"),
        "conf_threshold": test_params.get("confidence_threshold", 0.5),
        "iou_threshold": test_params.get("iou_threshold", 0.5),
        "max_samples": test_params.get("max_samples"),
    }


def load_samples(
    dataset_config: dict, max_samples: int | None = None
) -> Iterator[dict]:
    """Stream dataset samples using asqi.datasets.load_hf_dataset."""
    dataset_config.setdefault("loader_params", {})["streaming"] = True
    dataset = load_hf_dataset(dataset_config)
    for i, sample in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        yield sample


def call_detection_api(
    client: InferenceClient, image: Image.Image, model: str, api_key: str | None = None
) -> list:
    """Call HF Inference API for object detection."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    if model.startswith("http"):
        headers = {"Content-Type": "image/png"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = http_requests.post(
            model, headers=headers, data=buf.getvalue(), timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    return client.object_detection(image=buf.getvalue(), model=model)


def to_tensors(boxes: list, labels: list, scores: list | None = None) -> dict:
    """Convert lists to torchmetrics tensor format."""
    result = {
        "boxes": torch.tensor(boxes, dtype=torch.float32)
        if boxes
        else torch.zeros((0, 4)),
        "labels": torch.tensor(labels, dtype=torch.int64)
        if labels
        else torch.zeros(0, dtype=torch.int64),
    }
    if scores is not None:
        result["scores"] = (
            torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
        )
    return result


def parse_detections(detections: list, threshold: float, label_map: dict) -> dict:
    """Convert API detections to tensor format."""
    boxes, scores, labels = [], [], []
    for det in detections:
        if det["score"] >= threshold:
            box = det["box"]
            boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            scores.append(det["score"])
            labels.append(label_map.get(det["label"].lower(), -1))
    return to_tensors(boxes, labels, scores)


def parse_ground_truth(sample: dict, bbox_format: str) -> dict:
    """Convert ground truth annotations to tensor format."""
    objects = sample.get("objects", {})
    boxes = []
    for box in objects.get("bbox", []):
        if len(box) == 4:
            if bbox_format == "xyxy":
                boxes.append(box)
            else:  # xywh format
                x, y, w, h = box
                boxes.append([x, y, x + w, y + h])
    return to_tensors(boxes, objects.get("category", []))


_EXPORTED_METRICS = frozenset(
    {
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
    }
)


def calculate_metrics(predictions: list, targets: list, iou_threshold: float) -> dict:
    """Calculate mAP metrics. Returns only metrics listed in _EXPORTED_METRICS."""
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
    metric.update(predictions, targets)
    results = metric.compute()
    extracted = {}
    for k in sorted(_EXPORTED_METRICS):
        v = results.get(k)
        if v is None:
            log("WARN", f"Expected metric '{k}' missing from torchmetrics output")
            continue
        extracted[k] = float(v)
    return extracted


def format_value(value) -> str:
    """Format value for display."""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_report(metrics: dict, model: str, dataset: str) -> dict | None:
    """Write HTML report to output volume. Returns None if OUTPUT_MOUNT_PATH not set."""
    output_path = os.environ.get("OUTPUT_MOUNT_PATH")
    if not output_path:
        log("WARN", "OUTPUT_MOUNT_PATH not set, skipping report")
        return None

    reports_dir = Path(output_path) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    safe_model = html.escape(model)
    safe_dataset = html.escape(dataset)
    rows = "".join(
        f"<tr><td>{k}</td><td>{format_value(v)}</td></tr>" for k, v in metrics.items()
    )
    content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Detection Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; }}
        td, th {{ border: 1px solid #ccc; padding: 8px; }}
    </style>
</head>
<body>
    <h1>HF Vision Evaluator</h1>
    <p>Model: {safe_model} | Dataset: {safe_dataset}</p>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {rows}
    </table>
</body>
</html>"""

    report_path = reports_dir / "detection_report.html"
    report_path.write_text(content)
    log("INFO", f"Report written: {report_path}")

    return {
        "report_name": "detection_report",
        "report_type": "html",
        "report_path": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description="HF Vision Evaluator")
    parser.add_argument("--systems-params", required=True)
    parser.add_argument("--test-params", required=True)
    args = parser.parse_args()

    try:
        config = parse_config(
            json.loads(args.systems_params), json.loads(args.test_params)
        )
        log("INFO", f"Model: {config['model_id']}, Timeout: {config['timeout']}s")

        model_id = config["model_id"]
        if model_id.startswith("http"):
            client = InferenceClient(
                model=model_id,
                token=config["api_key"],
                timeout=config["timeout"],
            )
        else:
            client = InferenceClient(
                provider="hf-inference",
                api_key=config["api_key"],
                timeout=config["timeout"],
            )

        label_map = normalize_label_map(config["label_map"])
        predictions, targets = [], []

        for i, sample in enumerate(
            load_samples(config["ds_config"], config["max_samples"]), start=1
        ):
            image = sample.get("image")
            if not isinstance(image, Image.Image):
                continue

            log("INFO", f"[{i}] Calling API...")
            try:
                detections = call_detection_api(
                    client, image, config["model_id"], config["api_key"]
                )
                log("INFO", f"[{i}] Got {len(detections)} detections")
            except Exception as e:
                log("WARN", f"[{i}] API error: {e}")
                detections = []

            predictions.append(
                parse_detections(detections, config["conf_threshold"], label_map)
            )
            targets.append(parse_ground_truth(sample, config["bbox_format"]))

        if not predictions:
            raise ValueError("No valid samples to evaluate")

        metrics = calculate_metrics(predictions, targets, config["iou_threshold"])
        log(
            "INFO",
            f"Results: {', '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items()))}",
        )

        report = write_report(metrics, config["model_id"], "evaluation_data")

        output = {
            "test_results": {"success": True, "score": metrics["map"], **metrics},
            "generated_reports": [report] if report else [],
        }
        print(json.dumps(output, indent=2))

    except Exception as e:
        log("ERROR", str(e))
        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps(
                {"test_results": {"success": False, "error": str(e), "score": 0.0}},
                indent=2,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
