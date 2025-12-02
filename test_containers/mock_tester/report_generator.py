import os
from pathlib import Path


def write_mock_report(score: float, delay_used: int, base_url: str, model: str) -> str:
    """
    Generate a simple HTML technical report and save it into the mounted
    OUTPUT_MOUNT_PATH volume. Returns the full path to the report file.
    Does NOT affect the JSON printed by entrypoint.py.
    """
    output_root = Path(os.environ.get("OUTPUT_MOUNT_PATH"))
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "mock_tester_report.html"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Mock Tester Technical Report</title>
</head>
<body>
  <h1>Mock Tester – Technical Report</h1>

  <h2>Run Summary</h2>
  <ul>
    <li><strong>Score:</strong> {score:.3f}</li>
    <li><strong>Delay used (s):</strong> {delay_used}</li>
    <li><strong>Base URL:</strong> <code>{base_url}</code></li>
    <li><strong>Model:</strong> <code>{model}</code></li>
  </ul>

  <h2>Factors Explaining the Outcome</h2>
  <p>This is a mock container, so the score is random. In a real test, this
     section would summarize which metrics or conditions contributed to the
     final indicator level.</p>

  <h2>Actionable Insights</h2>
  <ul>
    <li>Reduce simulated delay (<code>delay_seconds</code>) to model better latency.</li>
    <li>Compare runs across different models to understand sensitivity.</li>
  </ul>

  <h2>Failures & Edge Cases</h2>
  <p>This demo does not generate failures, but a real container would list
     problematic requests or conversations here.</p>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return str(report_path)
