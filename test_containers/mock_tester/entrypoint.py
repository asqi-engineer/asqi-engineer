import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock test container")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        # Extract system_under_test
        test_id = test_params.get("id", "mock_test_run")
        sut_params = systems_params.get("system_under_test", {})
        if not sut_params:
            raise ValueError("Missing system_under_test in systems_params")

        # Validate SUT type
        sut_type = sut_params.get("type")
        if sut_type not in ["llm_api"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        # Extract SUT parameters (flattened structure)
        base_url = sut_params["base_url"]  # Required, validated upstream
        model = sut_params["model"]  # Required, validated upstream

        # Extract delay parameter
        delay_seconds = test_params.get("delay_seconds", 0)

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Simulate LLM API call (this is a mock, so we don't actually call the API)
        # In a real test container, you would use:
        # import openai
        # client = openai.OpenAI(base_url=base_url, api_key=os.getenv("API_KEY"))
        # response = client.chat.completions.create(model=model, messages=[...])

        # Always succeed with a random score

        metrics_data = {
            "success": True,
            "score": random.uniform(0.7, 1.0),
            "delay_used": delay_seconds,
            "base_url": base_url,
            "model": model,
        }

        output = {
            "test_results": metrics_data,
            "technical_reports": [write_quick_summary_report(test_id, metrics_data)],
        }
        # Output results as JSON
        print(json.dumps(output, indent=2))

        # Exit with appropriate code
        sys.exit(0 if output["test_results"]["success"] else 1)

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def write_quick_summary_report(test_id, metrics):
    output_root = Path(os.environ["OUTPUT_MOUNT_PATH"])
    output_root.mkdir(parents=True, exist_ok=True)
    filename = "quick_summary.html"
    file_path = output_root / "reports" / filename

    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
        subplot_titles=("Data Table", "Visual Chart"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=list(df.columns), fill_color="lightgrey", align="left"),
            cells=dict(values=[df["Metric"], df["Value"]], align="left"),
        ),
        row=1,
        col=1,
    )

    df_numeric = df.copy()
    df_numeric["Value"] = pd.to_numeric(df_numeric["Value"], errors="coerce")
    df_numeric = df_numeric.dropna()

    fig.add_trace(
        go.Bar(x=df_numeric["Metric"], y=df_numeric["Value"], name="Values"),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800, title_text=f"Test Report: {test_id}", showlegend=False
    )
    fig.write_html(file_path)
    return {
        "report_name": "quick_summary",
        "report_type": "html",
        "report_path": str(file_path),
    }


if __name__ == "__main__":
    main()
