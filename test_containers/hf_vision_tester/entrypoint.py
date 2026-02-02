"""HuggingFace Vision Evaluator - Object Detection via HF Inference API."""
import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import InferenceClient
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def log(level: str, message: str) -> None:
    """Log message to stderr."""
    print(f"[{level}] {message}", file=sys.stderr)


def parse_config(sys_params: dict, test_params: dict) -> dict:
    """Extract configuration from system and test parameters."""
    sut = sys_params.get("system_under_test", {})
    if sut.get("type") != "hf_inference_api":
        raise ValueError(f"Expected hf_inference_api, got: {sut.get('type')}")

    # Support both input_datasets format and direct params
    ds_config = test_params.get("input_datasets", {}).get("evaluation_data", {})
    loader = ds_config.get("loader_params", {})

    return {
        "model_id": sut.get("model_id", "facebook/detr-resnet-50"),
        "api_key": sut.get("api_key") or os.environ.get("HF_TOKEN"),
        "timeout": sut.get("timeout", 30.0),
        "dataset": loader.get("path") or test_params.get("dataset_path", "detection-datasets/coco"),
        "split": loader.get("split") or test_params.get("dataset_split", "val"),
        "label_map": ds_config.get("label_map") or test_params.get("label_map", {}),
        "bbox_format": ds_config.get("bbox_format") or test_params.get("bbox_format", "xyxy"),
        "conf_threshold": test_params.get("confidence_threshold", 0.5),
        "iou_threshold": test_params.get("iou_threshold", 0.5),
        "max_samples": test_params.get("max_samples"),
    }


def load_samples(path: str, split: str, max_samples: int | None) -> list:
    """Load dataset samples, streaming if max_samples is set."""
    log("INFO", f"Loading: {path} ({split})")
    try:
        if max_samples:
            log("INFO", f"Streaming {max_samples} samples...")
            stream = load_dataset(path, split=split, streaming=True)
            samples = []
            for i, s in enumerate(stream):
                if i >= max_samples:
                    break
                samples.append(s)
                log("INFO", f"  Loaded sample {i+1}/{max_samples}")
            return samples
        return list(load_dataset(path, split=split))
    except Exception as e:
        log("ERROR", f"Dataset loading failed: {e}")
        raise


def call_detection_api(client: InferenceClient, image: Image.Image, model: str) -> list:
    """Call HF Inference API for object detection."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f, format="JPEG")
        temp_path = f.name
    try:
        return client.object_detection(image=temp_path, model=model)
    finally:
        os.unlink(temp_path)


def to_tensors(boxes: list, labels: list, scores: list | None = None) -> dict:
    """Convert lists to torchmetrics tensor format."""
    result = {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
    }
    if scores is not None:
        result["scores"] = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
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


def calculate_metrics(predictions: list, targets: list, iou_threshold: float) -> dict:
    """Calculate mAP metrics."""
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
    metric.update(predictions, targets)
    results = metric.compute()
    return {
        "map": float(results.get("map", 0)),
        "map_50": float(results.get("map_50", 0)),
    }


def format_value(value) -> str:
    """Format value for display."""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def write_report(metrics: dict, model: str, dataset: str) -> dict:
    """Write HTML report to output volume."""
    reports_dir = Path(os.environ["OUTPUT_MOUNT_PATH"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = "".join(f"<tr><td>{k}</td><td>{format_value(v)}</td></tr>" for k, v in metrics.items())
    html = f"""<!DOCTYPE html>
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
    <p>Model: {model} | Dataset: {dataset}</p>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {rows}
    </table>
</body>
</html>"""

    report_path = reports_dir / "detection_report.html"
    report_path.write_text(html)
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
        # Parse configuration
        config = parse_config(json.loads(args.systems_params), json.loads(args.test_params))
        log("INFO", f"Model: {config['model_id']}, Timeout: {config['timeout']}s")

        # Load dataset
        samples = load_samples(config["dataset"], config["split"], config["max_samples"])

        # Initialize client
        client = InferenceClient(
            provider="hf-inference",
            api_key=config["api_key"],
            timeout=config["timeout"]
        )

        # Build label name -> id mapping
        label_map = {str(v).lower(): int(k) for k, v in config["label_map"].items()}

        # Run inference
        log("INFO", f"Running on {len(samples)} samples...")
        predictions, targets = [], []

        for i, sample in enumerate(samples):
            image = sample.get("image")
            if not isinstance(image, Image.Image):
                continue

            log("INFO", f"[{i + 1}/{len(samples)}] Calling API...")
            try:
                detections = call_detection_api(client, image, config["model_id"])
                log("INFO", f"[{i + 1}/{len(samples)}] Got {len(detections)} detections")
            except Exception as e:
                log("WARN", f"[{i + 1}] API error: {e}")
                detections = []

            predictions.append(parse_detections(detections, config["conf_threshold"], label_map))
            targets.append(parse_ground_truth(sample, config["bbox_format"]))

        if not predictions:
            raise ValueError("No valid samples to evaluate")

        # Calculate metrics
        metrics = calculate_metrics(predictions, targets, config["iou_threshold"])
        log("INFO", f"Results: mAP={metrics['map']:.4f}, mAP@50={metrics['map_50']:.4f}")

        # Write report and output results
        report = write_report(metrics, config["model_id"], config["dataset"])

        output = {
            "test_results": {"success": True, "score": metrics["map"], **metrics},
            "generated_reports": [report],
        }
        print(json.dumps(output, indent=2))

    except Exception as e:
        log("ERROR", str(e))
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"test_results": {"success": False, "error": str(e), "score": 0.0}}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
